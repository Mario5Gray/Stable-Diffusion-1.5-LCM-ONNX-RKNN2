// src/utils/cache.js

/**
 * Cache abstraction layer - storage agnostic interface.
 * Implementations can use IndexedDB, memory, localStorage, etc.
 */

/* ============================================================================
 * CACHE KEY GENERATION
 * ========================================================================== */

/**
 * Generate a deterministic cache key from generation parameters.
 * Same params always produce the same key.
 *
 * @param {object} params - Generation parameters
 * @returns {string} Cache key
 */
export function generateCacheKey(params) {
  const {
    prompt,
    size,
    steps,
    cfg,
    seed,
    superresLevel = 0,
  } = params;

  // Normalize and sort for consistency
  const normalized = {
    p: String(prompt || '').trim().toLowerCase(),
    sz: String(size || '512x512'),
    st: Number(steps) || 0,
    cfg: Number(cfg) || 0,
    sd: Number(seed) || 0,
    sr: Number(superresLevel) || 0,
  };

  // Simple deterministic hash
  const str = JSON.stringify(normalized);
  return hashString(str);
}

/**
 * Simple string hash (djb2 algorithm).
 * @param {string} str
 * @returns {string} Hex hash
 */
function hashString(str) {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) ^ str.charCodeAt(i);
  }
  // Convert to unsigned 32-bit and then to hex
  return (hash >>> 0).toString(16).padStart(8, '0');
}

/* ============================================================================
 * CACHE INTERFACE (Abstract)
 * ========================================================================== */

/**
 * @typedef {object} CacheEntry
 * @property {string} key - Cache key
 * @property {Blob} blob - Image blob data
 * @property {object} metadata - Generation metadata
 * @property {number} createdAt - Timestamp
 * @property {number} accessedAt - Last access timestamp
 * @property {number} size - Blob size in bytes
 */

/**
 * @typedef {object} CacheInterface
 * @property {function(string): Promise<CacheEntry|null>} get - Get entry by key
 * @property {function(string, Blob, object): Promise<void>} set - Store entry
 * @property {function(string): Promise<boolean>} has - Check if key exists
 * @property {function(string): Promise<boolean>} delete - Delete entry
 * @property {function(): Promise<void>} clear - Clear all entries
 * @property {function(): Promise<number>} size - Get total entries count
 * @property {function(): Promise<number>} totalBytes - Get total storage used
 */

/* ============================================================================
 * INDEXEDDB IMPLEMENTATION
 * ========================================================================== */

const DB_NAME = 'lcm-image-cache';
const DB_VERSION = 1;
const STORE_NAME = 'images';

/**
 * Open IndexedDB connection.
 * @returns {Promise<IDBDatabase>}
 */
function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = event.target.result;

      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'key' });
        store.createIndex('createdAt', 'createdAt', { unique: false });
        store.createIndex('accessedAt', 'accessedAt', { unique: false });
        store.createIndex('size', 'size', { unique: false });
      }
    };
  });
}

/**
 * Create an IndexedDB-backed cache.
 *
 * @param {object} options
 * @param {number} [options.maxEntries=500] - Max entries before LRU eviction
 * @param {number} [options.maxBytes=500*1024*1024] - Max storage (500MB default)
 * @returns {CacheInterface}
 */
export function createIndexedDBCache(options = {}) {
  const {
    maxEntries = 500,
    maxBytes = 500 * 1024 * 1024, // 500MB
  } = options;

  let dbPromise = null;

  /**
   * Get database connection (lazy init).
   */
  const getDb = () => {
    if (!dbPromise) {
      dbPromise = openDatabase();
    }
    return dbPromise;
  };

  /**
   * Run a transaction and return result.
   */
  const withStore = async (mode, callback) => {
    const db = await getDb();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, mode);
      const store = tx.objectStore(STORE_NAME);

      let result;
      try {
        result = callback(store, tx);
      } catch (err) {
        reject(err);
        return;
      }

      tx.oncomplete = () => resolve(result);
      tx.onerror = () => reject(tx.error);
    });
  };

  /**
   * Promisify IDBRequest.
   */
  const promisify = (request) => {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  };

  return {
    /**
     * Get cache entry by key.
     * Updates accessedAt timestamp.
     */
    async get(key) {
      try {
        const db = await getDb();
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);

        const entry = await promisify(store.get(key));

        if (entry) {
          // Update access time
          entry.accessedAt = Date.now();
          store.put(entry);
        }

        return entry || null;
      } catch (err) {
        console.warn('[Cache] get failed:', err);
        return null;
      }
    },

    /**
     * Store blob with metadata.
     */
    async set(key, blob, metadata = {}) {
      try {
        const entry = {
          key,
          blob,
          metadata,
          createdAt: Date.now(),
          accessedAt: Date.now(),
          size: blob.size,
        };

        await withStore('readwrite', (store) => {
          store.put(entry);
        });

        // Trigger eviction check (async, don't await)
        this._evictIfNeeded();
      } catch (err) {
        console.warn('[Cache] set failed:', err);
      }
    },

    /**
     * Check if key exists.
     */
    async has(key) {
      try {
        const db = await getDb();
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const count = await promisify(store.count(key));
        return count > 0;
      } catch (err) {
        console.warn('[Cache] has failed:', err);
        return false;
      }
    },

    /**
     * Delete entry by key.
     */
    async delete(key) {
      try {
        await withStore('readwrite', (store) => {
          store.delete(key);
        });
        return true;
      } catch (err) {
        console.warn('[Cache] delete failed:', err);
        return false;
      }
    },

    /**
     * Clear all entries.
     */
    async clear() {
      try {
        await withStore('readwrite', (store) => {
          store.clear();
        });
      } catch (err) {
        console.warn('[Cache] clear failed:', err);
      }
    },

    /**
     * Get total entry count.
     */
    async size() {
      try {
        const db = await getDb();
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        return await promisify(store.count());
      } catch (err) {
        console.warn('[Cache] size failed:', err);
        return 0;
      }
    },

    /**
     * Get total bytes used.
     */
    async totalBytes() {
      try {
        const db = await getDb();
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);

        let total = 0;
        const cursor = store.openCursor();

        return new Promise((resolve, reject) => {
          cursor.onsuccess = (event) => {
            const c = event.target.result;
            if (c) {
              total += c.value.size || 0;
              c.continue();
            } else {
              resolve(total);
            }
          };
          cursor.onerror = () => reject(cursor.error);
        });
      } catch (err) {
        console.warn('[Cache] totalBytes failed:', err);
        return 0;
      }
    },

    /**
     * Get cache statistics.
     */
    async stats() {
      const [count, bytes] = await Promise.all([
        this.size(),
        this.totalBytes(),
      ]);
      return {
        entries: count,
        bytes,
        maxEntries,
        maxBytes,
        utilizationEntries: count / maxEntries,
        utilizationBytes: bytes / maxBytes,
      };
    },

    /**
     * Evict oldest entries if over limits.
     * Uses LRU (least recently accessed).
     */
    async _evictIfNeeded() {
      try {
        const [count, bytes] = await Promise.all([
          this.size(),
          this.totalBytes(),
        ]);

        const needsEviction = count > maxEntries || bytes > maxBytes;
        if (!needsEviction) return;

        const db = await getDb();
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const index = store.index('accessedAt');

        // Get oldest entries
        const toDelete = Math.max(
          count - maxEntries + 10, // Evict 10 extra to avoid frequent eviction
          Math.ceil((bytes - maxBytes) / (bytes / count)) + 5
        );

        let deleted = 0;
        const cursor = index.openCursor();

        return new Promise((resolve) => {
          cursor.onsuccess = (event) => {
            const c = event.target.result;
            if (c && deleted < toDelete) {
              store.delete(c.primaryKey);
              deleted++;
              c.continue();
            } else {
              console.log(`[Cache] Evicted ${deleted} entries`);
              resolve();
            }
          };
          cursor.onerror = () => resolve();
        });
      } catch (err) {
        console.warn('[Cache] eviction failed:', err);
      }
    },

    /**
     * Close database connection.
     */
    async close() {
      if (dbPromise) {
        const db = await dbPromise;
        db.close();
        dbPromise = null;
      }
    },
  };
}

/* ============================================================================
 * IN-MEMORY CACHE (Fallback / Testing)
 * ========================================================================== */

/**
 * Create an in-memory cache (for testing or fallback).
 * Data is lost on page refresh.
 *
 * @param {object} options
 * @param {number} [options.maxEntries=100] - Max entries
 * @returns {CacheInterface}
 */
export function createMemoryCache(options = {}) {
  const { maxEntries = 100 } = options;
  const store = new Map();

  return {
    async get(key) {
      const entry = store.get(key);
      if (entry) {
        entry.accessedAt = Date.now();
      }
      return entry || null;
    },

    async set(key, blob, metadata = {}) {
      const entry = {
        key,
        blob,
        metadata,
        createdAt: Date.now(),
        accessedAt: Date.now(),
        size: blob.size,
      };
      store.set(key, entry);

      // Simple LRU: delete oldest if over limit
      if (store.size > maxEntries) {
        const oldest = [...store.entries()]
          .sort((a, b) => a[1].accessedAt - b[1].accessedAt)[0];
        if (oldest) store.delete(oldest[0]);
      }
    },

    async has(key) {
      return store.has(key);
    },

    async delete(key) {
      return store.delete(key);
    },

    async clear() {
      store.clear();
    },

    async size() {
      return store.size;
    },

    async totalBytes() {
      let total = 0;
      for (const entry of store.values()) {
        total += entry.size || 0;
      }
      return total;
    },

    async stats() {
      const bytes = await this.totalBytes();
      return {
        entries: store.size,
        bytes,
        maxEntries,
        utilizationEntries: store.size / maxEntries,
      };
    },

    async close() {
      // No-op for memory cache
    },
  };
}

/* ============================================================================
 * CACHE FACTORY
 * ========================================================================== */

/**
 * Create the appropriate cache based on environment.
 * Falls back to memory cache if IndexedDB unavailable.
 *
 * @param {object} options - Cache options
 * @returns {CacheInterface}
 */
export function createCache(options = {}) {
  // Check if IndexedDB is available
  if (typeof indexedDB !== 'undefined') {
    try {
      return createIndexedDBCache(options);
    } catch (err) {
      console.warn('[Cache] IndexedDB unavailable, using memory cache:', err);
    }
  }

  return createMemoryCache(options);
}
