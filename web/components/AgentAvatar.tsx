// Avatar de l'agent conversationnel : une tête stylisée avec casque (couleurs CGECI).
export default function AgentAvatar({ size = 32 }: { size?: number }) {
  const id = "agrad" + size;
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none"
         xmlns="http://www.w3.org/2000/svg" aria-label="Assistant">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#6b8cff" />
          <stop offset="1" stopColor="#3a5bbf" />
        </linearGradient>
      </defs>
      {/* arceau du casque */}
      <path d="M15 33 a17 17 0 0 1 34 0" fill="none" stroke="#1a3a5c"
            strokeWidth="4" strokeLinecap="round" />
      {/* écouteurs */}
      <rect x="9" y="30" width="7.5" height="13" rx="3.75" fill="#1a3a5c" />
      <rect x="47.5" y="30" width="7.5" height="13" rx="3.75" fill="#1a3a5c" />
      {/* tête */}
      <rect x="16" y="20" width="32" height="31" rx="13" fill={`url(#${id})`} />
      {/* yeux */}
      <circle cx="26" cy="34" r="3.1" fill="#fff" />
      <circle cx="38" cy="34" r="3.1" fill="#fff" />
      {/* sourire */}
      <path d="M26.5 41.5 q5.5 4.5 11 0" fill="none" stroke="#fff"
            strokeWidth="2.4" strokeLinecap="round" />
      {/* micro */}
      <path d="M12.5 42 q-1.5 8 8 8.5" fill="none" stroke="#1a3a5c"
            strokeWidth="3" strokeLinecap="round" />
      <circle cx="20.5" cy="50.5" r="2.7" fill="#e67e22" />
    </svg>
  );
}
