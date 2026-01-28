// src/components/chat/MessageComposer.jsx

import React from 'react';
import { Textarea } from '@/components/ui/textarea';
import { Button } from '@/components/ui/button';
import { Send, Square } from 'lucide-react';

/**
 * Message composer component - textarea + send/cancel buttons.
 */
export function MessageComposer({
  prompt,
  onPromptChange,
  onSend,
  onCancelAll,
  onKeyDown,
  onFocus,
  inflightCount,
  disabled,
  currentParams,
  serverLabel,
}) {
  console.log('MessageComposer rendering', { prompt, disabled }); // DEBUG

  return (
    <div className="p-2 md:p-2 option-panel-area">
      <div className="flex items-center rounded-base bg-neutral-secondary-soft">
        <Textarea
          value={prompt}
          onChange={(e) => onPromptChange(e.target.value)}
          onKeyDown={onKeyDown}
          onFocus={onFocus}
          placeholder="Describe what you want to generate…"
          className="mx-4 bg-neutral-primary-medium border border-default-medium text-heading text-sm rounded-base focus:ring-brand focus:border-brand block w-full px-3 py-2.5 placeholder:text-body"
        />
        <div className="flex flex-col mt-2 gap-2">
  
          <Button
            style={{ background: 'linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%)' }}
            onClick={onSend}
            disabled={disabled}
            className="
              relative overflow-hidden
              border border-purple-400/40
              bg-gradient-to-br from-purple-500/90 to-pink-500/90
              text-white
              shadow-md
              hover:from-purple-500 hover:to-pink-500
              active:scale-[0.75]
              transition-all
            "
          >
            
            Send
          </Button>
        </div>
      </div>

      {/* Current params summary */}
{/*      {currentParams && (
        <div className="mt-2 text-xs text-muted-foreground">
          Current: size {currentParams.size} · steps {currentParams.steps} · CFG{' '}
          {Number(currentParams.cfg).toFixed(1)} · seed{' '}
          {currentParams.seedMode === 'random' ? 'random' : currentParams.seed}
          {currentParams.superresLevel > 0
            ? ` · SR ${currentParams.superresLevel}`
            : ''}{' '}
          · {serverLabel}
        </div>
      )}*/}
    </div>
  );
}