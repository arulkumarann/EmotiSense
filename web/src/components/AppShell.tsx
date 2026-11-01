import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import { NavLink, useLocation, useOutlet } from "react-router-dom";
import { useHealth } from "../hooks/useData";
import { Atmosphere, Logo, ThemeToggle } from "./Chrome";
import { IconBook, IconChart, IconLibrary, IconLive, IconLock, IconSliders, IconWave } from "./Icons";

const NAV = [
  { to: "/app", label: "Analyze", icon: IconWave, end: true },
  { to: "/app/library", label: "Library", icon: IconLibrary },
  { to: "/app/insights", label: "Insights", icon: IconChart },
  { to: "/app/live", label: "Live replay", icon: IconLive },
  { to: "/app/settings", label: "Settings", icon: IconSliders },
  { to: "/app/method", label: "Method", icon: IconBook },
];

function StatusPill() {
  const { health, offline } = useHealth();
  const state = offline ? "bad" : health?.model === "ready" ? "ready" : health?.model === "error" || health?.model === "absent" ? "bad" : "loading";
  const text = offline ? "Server offline" : health?.model === "ready" ? "Model ready" : health?.model === "error" ? "Model error" : health?.model === "absent" ? "No model" : "Warming up";
  return (
    <span className="status-pill" title={health?.model_error ?? undefined}>
      <span className={`dot ${state}`} />
      {text}
    </span>
  );
}

export default function AppShell() {
  const location = useLocation();
  const outlet = useOutlet();
  return (
    <div className="app">
      <Atmosphere />
      <aside className="rail">
        <div className="rail-inner glass">
          <Logo to="/" />
          <nav className="rail-nav" aria-label="Main">
            <LayoutGroup id="nav">
              {NAV.map(({ to, label, icon: Icon, end }) => (
                <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
                  {({ isActive }) => (
                    <>
                      {isActive && <motion.span layoutId="nav-pill" className="nav-pill" transition={{ type: "spring", stiffness: 420, damping: 36 }} />}
                      <Icon />
                      <span>{label}</span>
                    </>
                  )}
                </NavLink>
              ))}
            </LayoutGroup>
          </nav>
          <div className="rail-foot">
            <div className="privacy-note">
              <IconLock />
              <span>Everything runs on this computer. Your audio never leaves it.</span>
            </div>
          </div>
        </div>
      </aside>
      <main className="main">
        <div className="topbar">
          <StatusPill />
          <ThemeToggle />
        </div>
        <AnimatePresence mode="wait" initial={false}>
          <div key={location.pathname} style={{ display: "contents" }}>
            {outlet}
          </div>
        </AnimatePresence>
      </main>
    </div>
  );
}
