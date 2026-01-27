module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
    "./**/*.{js,jsx,ts,tsx}",   // 🔥 add this
  ],
  theme: { extend: { /* ... */ } },
  plugins: [],
//  plugins: [require("daisyui")],
};
