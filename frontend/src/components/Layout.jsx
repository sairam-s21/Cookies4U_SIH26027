import { NavLink, Outlet, useLocation } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", end: true, icon: "▦", sub: "Live overview — MAS ⟶ JTJ corridor" },
  { to: "/corridor-map", label: "Corridor Map", icon: "⬢", sub: "Maintenance blocks by time, live" },
  { to: "/waiting-list", label: "Waiting List", icon: "☰", sub: "Requests awaiting sign-off" },
  { to: "/recommended-scheduling", label: "Recommended Scheduling", icon: "✦", sub: "AI-generated schedule options" },
  { to: "/schedule", label: "Weekly / Monthly Schedule", icon: "▤", sub: "Corridor occupancy matrix" },
  { to: "/approved-tasks", label: "Approved Tasks", icon: "↺", sub: "Approved maintenance blocks" },
  { to: "/completed-history", label: "Completed History", icon: "◷", sub: "Granted blocks already completed" },
  { to: "/emergency-handling", label: "Emergency Handling", icon: "⚠", sub: "Create and resolve emergency situations" },
];

export default function Layout() {
  const location = useLocation();
  const current =
    NAV_ITEMS.find((item) => (item.end ? location.pathname === item.to : location.pathname.startsWith(item.to))) ||
    NAV_ITEMS[0];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">RC</div>
          <div>
            <h1>RailBlock Co-Pilot</h1>
            <div className="sub">MAS–JTJ · Southern Railway</div>
          </div>
        </div>
        <nav>
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
            >
              <span className="ic">{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          PS 26027 — AI-Powered Automatic Block Planning
        </div>
      </aside>

      <div className="main">
        <div className="topbar">
          <div className="topbar-left">
            <h2>{current.label}</h2>
            <div className="sub">{current.sub}</div>
          </div>
          <div className="topbar-right">
            <select className="div-select" defaultValue="MAS-JTJ">
              <option value="MAS-JTJ">MAS–JTJ Corridor</option>
            </select>
            <div className="avatar">SR</div>
          </div>
        </div>

        <div className="content">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
