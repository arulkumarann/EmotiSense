import { motion, useInView, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useTheme } from "../hooks/useTheme";
import { IconMoon, IconSun } from "./Icons";

export const EASE = [0.22, 1, 0.36, 1] as const;

/** Slowly drifting blurred colour fields behind everything. Pure CSS, GPU-composited. */
export function Atmosphere() {
  return (
    <div className="atmosphere" aria-hidden="true">
      <div className="blob b1" />
      <div className="blob b2" />
      <div className="blob b3" />
    </div>
  );
}

export function Logo({ to = "/", size = 30 }: { to?: string; size?: number }) {
  return (
    <Link to={to} className="logo" aria-label="EmotiSense home">
      <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
        <defs>
          <linearGradient id="logo-g" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="var(--accent-2)" />
            <stop offset="1" stopColor="var(--accent)" />
          </linearGradient>
        </defs>
        <rect width="64" height="64" rx="19" fill="var(--surface-solid)" stroke="var(--line-strong)" />
        <path d="M11 35C18 14 24 14 30 33S44 52 53 25" fill="none" stroke="url(#logo-g)" strokeWidth="6" strokeLinecap="round" />
      </svg>
      <span className="logo-word">
        Emoti<em>Sense</em>
      </span>
    </Link>
  );
}

export function ThemeToggle() {
  const { resolved, setTheme } = useTheme();
  return (
    <button
      className="icon-btn theme-toggle"
      aria-label={resolved === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        setTheme(resolved === "dark" ? "light" : "dark", { x: r.left + r.width / 2, y: r.top + r.height / 2 });
      }}
    >
      <motion.span key={resolved} initial={{ rotate: -80, opacity: 0, scale: 0.6 }} animate={{ rotate: 0, opacity: 1, scale: 1 }} transition={{ duration: 0.6, ease: EASE }} style={{ display: "grid" }}>
        {resolved === "dark" ? <IconSun /> : <IconMoon />}
      </motion.span>
    </button>
  );
}

/** Fades, lifts and un-blurs its children as they scroll into view. */
export function Reveal({ children, delay = 0, y = 28, className, style, as = "div" }: {
  children: ReactNode; delay?: number; y?: number; className?: string; style?: CSSProperties; as?: "div" | "section" | "li" | "article";
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "0px 0px -12% 0px" });
  const reduced = useReducedMotion();
  const Tag = motion[as] as typeof motion.div;
  return (
    <Tag
      ref={ref}
      className={className}
      style={style}
      initial={reduced ? false : { opacity: 0, y, filter: "blur(12px)" }}
      animate={inView || reduced ? { opacity: 1, y: 0, filter: "blur(0px)" } : undefined}
      transition={{ duration: 0.9, delay, ease: EASE }}
    >
      {children}
    </Tag>
  );
}

/** Route-level enter/exit: opacity, lift and blur. */
export function PageTransition({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 18, filter: "blur(12px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      exit={{ opacity: 0, y: -10, filter: "blur(8px)" }}
      transition={{ duration: 0.55, ease: EASE }}
    >
      {children}
    </motion.div>
  );
}

/** Counts up to a value once visible. */
export function CountUp({ to, decimals = 0, suffix = "", prefix = "" }: { to: number; decimals?: number; suffix?: string; prefix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const reduced = useReducedMotion();
  const [value, setValue] = useState(reduced ? to : 0);
  useEffect(() => {
    if (!inView || reduced) return;
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / 1600);
      setValue(to * (1 - Math.pow(1 - p, 4)));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, to, reduced]);
  return <span ref={ref} className="tabular">{prefix}{value.toFixed(decimals)}{suffix}</span>;
}
