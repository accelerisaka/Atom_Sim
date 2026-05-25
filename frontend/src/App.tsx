import { Link, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage";
import ScenarioPage from "./pages/ScenarioPage";

export default function App() {
  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand">
          <span className="brand-dot" />
          Atom_Sim · 联合仿真可视化
        </Link>
        <div className="topbar-spacer" />
        <a
          className="topbar-link"
          href="https://github.com"
          onClick={(e) => e.preventDefault()}
        >
          v0.1
        </a>
      </header>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/scenario/:name" element={<ScenarioPage />} />
      </Routes>
    </div>
  );
}
