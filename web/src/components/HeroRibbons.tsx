import { useEffect, useRef } from "react";
import { useTheme } from "../hooks/useTheme";

/** Flowing, glowing sound ribbons. Canvas 2D, capped DPR, pauses when off-screen or reduced-motion is on. */
export function HeroRibbons({ className }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { resolved } = useTheme();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const ctx = canvas.getContext("2d");
    if (!ctx) return undefined;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const dark = resolved === "dark";
    const style = getComputedStyle(document.documentElement);
    const a1 = style.getPropertyValue("--accent").trim() || "#3b74ff";
    const a2 = style.getPropertyValue("--accent-2").trim() || "#8db2ff";

    let w = 0, h = 0, raf = 0, visible = true;
    const mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5 };
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      w = rect.width; h = rect.height;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    const io = new IntersectionObserver((e) => { visible = e[0].isIntersecting; });
    io.observe(canvas);
    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      mouse.tx = (e.clientX - r.left) / r.width;
      mouse.ty = (e.clientY - r.top) / r.height;
    };
    window.addEventListener("pointermove", onMove, { passive: true });

    const RIBBONS = 7;
    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      if (!visible || document.hidden) return;
      mouse.x += (mouse.tx - mouse.x) * 0.05;
      mouse.y += (mouse.ty - mouse.y) * 0.05;
      const t = reduce ? 0 : now / 1000;
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = dark ? "lighter" : "source-over";
      for (let i = 0; i < RIBBONS; i++) {
        const k = i / (RIBBONS - 1);
        const grad = ctx.createLinearGradient(0, 0, w, 0);
        grad.addColorStop(0, "rgba(0,0,0,0)");
        grad.addColorStop(0.25, a1);
        grad.addColorStop(0.7, a2);
        grad.addColorStop(1, "rgba(0,0,0,0)");
        ctx.strokeStyle = grad;
        ctx.lineWidth = i === 3 ? 2.2 : 1.1;
        ctx.globalAlpha = (dark ? 0.55 : 0.5) * (0.35 + 0.65 * (1 - Math.abs(k - 0.5) * 1.6));
        ctx.shadowColor = a1;
        ctx.shadowBlur = i === 3 ? 22 : 10;
        ctx.beginPath();
        const steps = 90;
        for (let s = 0; s <= steps; s++) {
          const u = s / steps;
          const x = u * w;
          const envelope = Math.sin(Math.PI * u) ** 1.4;
          const phase = t * (0.5 + k * 0.28) + k * 2.2;
          const wob = Math.sin(u * 7.5 + phase) * 0.55 + Math.sin(u * 13 - phase * 1.3 + k) * 0.28 + Math.sin(u * 3.2 + phase * 0.7) * 0.5;
          const spread = (k - 0.5) * 0.5 + (mouse.y - 0.5) * 0.12;
          const y = h * (0.5 + spread * envelope) + wob * h * 0.17 * envelope * (0.7 + 0.5 * mouse.x);
          if (s === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      ctx.shadowBlur = 0;
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      window.removeEventListener("pointermove", onMove);
    };
  }, [resolved]);

  return <canvas ref={ref} className={className} aria-hidden="true" />;
}
