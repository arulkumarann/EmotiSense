import type { ReactNode, SVGProps } from "react";

type P = SVGProps<SVGSVGElement>;
const base = (children: ReactNode, props: P) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>
    {children}
  </svg>
);

export const IconMic = (p: P) => base(<><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8.5 21h7" /></>, p);
export const IconUpload = (p: P) => base(<><path d="M12 16V4M7 9l5-5 5 5" /><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" /></>, p);
export const IconPlay = (p: P) => base(<path d="M7 4.5v15a.5.5 0 0 0 .77.42l11.4-7.5a.5.5 0 0 0 0-.84L7.77 4.08A.5.5 0 0 0 7 4.5Z" fill="currentColor" stroke="none" />, p);
export const IconPause = (p: P) => base(<><rect x="6" y="4.5" width="4" height="15" rx="1.2" fill="currentColor" stroke="none" /><rect x="14" y="4.5" width="4" height="15" rx="1.2" fill="currentColor" stroke="none" /></>, p);
export const IconBack = (p: P) => base(<path d="M11 6 5 12l6 6M19 6l-6 6 6 6" />, p);
export const IconForward = (p: P) => base(<path d="m13 6 6 6-6 6M5 6l6 6-6 6" />, p);
export const IconWave = (p: P) => base(<path d="M3 12h2M7 7v10M11 4v16M15 8v8M19 10v4M21 12h0" />, p);
export const IconLibrary = (p: P) => base(<><rect x="3.5" y="4" width="5" height="16" rx="1.4" /><rect x="10.5" y="4" width="5" height="16" rx="1.4" /><path d="m17.5 6.5 3.2-.7 2 13.5-3.2.7z" transform="translate(-1.4 0)" /></>, p);
export const IconChart = (p: P) => base(<><path d="M4 20V8M10 20V4M16 20v-9M22 20H2" /></>, p);
export const IconLive = (p: P) => base(<><circle cx="12" cy="12" r="2.2" /><path d="M7.8 7.8a6 6 0 0 0 0 8.4M16.2 7.8a6 6 0 0 1 0 8.4M4.9 4.9a10 10 0 0 0 0 14.2M19.1 4.9a10 10 0 0 1 0 14.2" /></>, p);
export const IconSliders = (p: P) => base(<><path d="M4 7h9M17 7h3M4 17h3M11 17h9" /><circle cx="15" cy="7" r="2" /><circle cx="9" cy="17" r="2" /></>, p);
export const IconBook = (p: P) => base(<><path d="M5 4.5A1.5 1.5 0 0 1 6.5 3H19v15H6.5A1.5 1.5 0 0 0 5 19.5z" /><path d="M5 19.5A1.5 1.5 0 0 0 6.5 21H19v-3" /></>, p);
export const IconSun = (p: P) => base(<><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>, p);
export const IconMoon = (p: P) => base(<path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5a8.5 8.5 0 1 0 10.7 10.7Z" />, p);
export const IconTrash = (p: P) => base(<><path d="M4 7h16M10 11v6M14 11v6" /><path d="M6 7l1 12a1 1 0 0 0 1 .9h8a1 1 0 0 0 1-.9l1-12M9 7V4h6v3" /></>, p);
export const IconPlus = (p: P) => base(<path d="M12 5v14M5 12h14" />, p);
export const IconCheck = (p: P) => base(<path d="m5 12.5 4.5 4.5L19 7" />, p);
export const IconClose = (p: P) => base(<path d="M6 6l12 12M18 6 6 18" />, p);
export const IconArrow = (p: P) => base(<path d="M5 12h14m-6-6 6 6-6 6" className="arrow" />, { ...p, className: `arrow ${p.className ?? ""}` });
export const IconNote = (p: P) => base(<><path d="M5 4h14v12l-5 5H5z" /><path d="M14 21v-5h5M8 9h8M8 13h4" /></>, p);
export const IconFlag = (p: P) => base(<path d="M5 21V4m0 0h11l-2 4 2 4H5" />, p);
export const IconSpark = (p: P) => base(<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8z" />, p);
export const IconShield = (p: P) => base(<><path d="M12 3 4.5 6v6c0 4.4 3.1 7.7 7.5 9 4.4-1.3 7.5-4.6 7.5-9V6z" /><path d="m9 12 2.2 2.2L15.5 10" /></>, p);
export const IconDownload = (p: P) => base(<><path d="M12 4v12M7 11l5 5 5-5" /><path d="M4 18v1a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-1" /></>, p);
export const IconVolume = (p: P) => base(<><path d="M4 10v4h3.5L12 18V6L7.5 10z" /><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11" /></>, p);
export const IconSearch = (p: P) => base(<><circle cx="11" cy="11" r="6.5" /><path d="m20 20-4-4" /></>, p);
export const IconRecord = (p: P) => base(<circle cx="12" cy="12" r="5.5" fill="currentColor" stroke="none" />, p);
export const IconStop = (p: P) => base(<rect x="6.5" y="6.5" width="11" height="11" rx="2.4" fill="currentColor" stroke="none" />, p);
export const IconLock = (p: P) => base(<><rect x="5" y="10.5" width="14" height="10" rx="2.5" /><path d="M8 10.5V8a4 4 0 0 1 8 0v2.5" /></>, p);
export const IconMenu = (p: P) => base(<path d="M4 7h16M4 12h16M4 17h10" />, p);
export const IconRepeat = (p: P) => base(<><path d="M4 11V9a3 3 0 0 1 3-3h13M16 3l4 3-4 3" /><path d="M20 13v2a3 3 0 0 1-3 3H4M8 21l-4-3 4-3" /></>, p);
