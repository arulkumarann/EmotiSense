import { Suspense, lazy, useEffect } from "react";
import { Link, Route, Routes, useLocation } from "react-router-dom";
import AppShell from "./components/AppShell";
import { Atmosphere } from "./components/Chrome";

const Landing = lazy(() => import("./pages/Landing"));
const Analyze = lazy(() => import("./pages/Analyze"));
const Session = lazy(() => import("./pages/Session"));
const Library = lazy(() => import("./pages/Library"));
const Insights = lazy(() => import("./pages/Insights"));
const Live = lazy(() => import("./pages/Live"));
const Settings = lazy(() => import("./pages/Settings"));
const Method = lazy(() => import("./pages/Method"));

function ScrollTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [pathname]);
  return null;
}

function NotFound() {
  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", textAlign: "center", gap: "1rem" }}>
      <Atmosphere />
      <div>
        <p className="eyebrow">404</p>
        <h1 className="display" style={{ fontSize: "4rem", margin: "0.6rem 0 1.4rem" }}>Lost the thread</h1>
        <Link className="btn btn-primary" to="/">Back home</Link>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <>
      <ScrollTop />
      <Suspense fallback={<Atmosphere />}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/app" element={<AppShell />}>
            <Route index element={<Analyze />} />
            <Route path="session/:id" element={<Session />} />
            <Route path="library" element={<Library />} />
            <Route path="insights" element={<Insights />} />
            <Route path="live" element={<Live />} />
            <Route path="settings" element={<Settings />} />
            <Route path="method" element={<Method />} />
          </Route>
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Suspense>
    </>
  );
}
