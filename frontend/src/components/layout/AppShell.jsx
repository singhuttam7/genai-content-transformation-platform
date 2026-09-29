import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";

import { navigation } from "../../constants/navigation";

const THEME_STORAGE_KEY = "genai-platform-theme";

function getInitialTheme() {
  if (typeof window === "undefined") {
    return "light";
  }

  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);

  if (storedTheme === "light" || storedTheme === "dark") {
    return storedTheme;
  }

  return window.matchMedia?.("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function AppShell({ children }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [theme, setTheme] = useState(getInitialTheme);

  const location = useLocation();

  const currentNavigationItem =
    navigation
      .flatMap((section) => section.items)
      .find((item) => item.path === location.pathname) ??
    navigation[0].items[0];

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    function handleEscape(event) {
      if (event.key === "Escape") {
        setMobileMenuOpen(false);
      }
    }

    window.addEventListener("keydown", handleEscape);

    return () => {
      window.removeEventListener("keydown", handleEscape);
    };
  }, []);

  function toggleTheme() {
    setTheme((currentTheme) => (currentTheme === "light" ? "dark" : "light"));
  }

  function toggleSidebar() {
    setSidebarCollapsed((current) => !current);
  }

  const sidebarClassName = [
    "app-sidebar",
    sidebarCollapsed ? "app-sidebar-collapsed" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={[
        "app-shell",
        sidebarCollapsed ? "app-shell-sidebar-collapsed" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {/* ============================================================
          Desktop Sidebar
          ============================================================ */}

      <aside className={sidebarClassName}>
        <div className="app-brand">
          <div className="app-brand-mark" aria-hidden="true">
            ✦
          </div>

          <div className="app-brand-copy">
            <div className="app-brand-name">GenAI Platform</div>
            <div className="app-brand-subtitle">Content Transformation</div>
          </div>
        </div>

        <button
          type="button"
          className="sidebar-collapse-button"
          onClick={toggleSidebar}
          aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <span aria-hidden="true">{sidebarCollapsed ? "›" : "‹"}</span>
        </button>

        <nav className="app-navigation" aria-label="Primary navigation">
          {navigation.map((section) => (
            <div className="navigation-section" key={section.section}>
              {!sidebarCollapsed && (
                <div className="navigation-section-title">
                  {section.section}
                </div>
              )}

              {section.items.map((item) => (
                <NavLink
                  key={item.id}
                  to={item.path}
                  end={item.path === "/"}
                  className={({ isActive }) =>
                    ["navigation-item", isActive ? "active" : ""]
                      .filter(Boolean)
                      .join(" ")
                  }
                  title={
                    sidebarCollapsed
                      ? (item.description ?? item.label)
                      : item.description
                  }
                >
                  <span className="navigation-item-icon" aria-hidden="true">
                    {getNavigationIcon(item.id)}
                  </span>

                  {!sidebarCollapsed && <span>{item.label}</span>}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        {!sidebarCollapsed && (
          <div className="sidebar-status">
            <span className="status-indicator" aria-hidden="true" />

            <div>
              <div className="sidebar-status-title">System operational</div>

              <div className="sidebar-status-subtitle">
                All core services available
              </div>
            </div>
          </div>
        )}
      </aside>

      {/* ============================================================
          Mobile Navigation
          ============================================================ */}

      {mobileMenuOpen && (
        <div
          className="mobile-navigation-overlay"
          onClick={() => setMobileMenuOpen(false)}
          role="presentation"
        >
          <aside
            className="mobile-sidebar"
            onClick={(event) => event.stopPropagation()}
            aria-label="Mobile navigation"
          >
            <div className="app-brand">
              <div className="app-brand-mark" aria-hidden="true">
                ✦
              </div>

              <div className="app-brand-copy">
                <div className="app-brand-name">GenAI Platform</div>

                <div className="app-brand-subtitle">Content Transformation</div>
              </div>

              <button
                type="button"
                className="mobile-sidebar-close"
                onClick={() => setMobileMenuOpen(false)}
                aria-label="Close navigation"
              >
                ×
              </button>
            </div>

            <nav
              className="app-navigation"
              aria-label="Mobile primary navigation"
            >
              {navigation.map((section) => (
                <div className="navigation-section" key={section.section}>
                  <div className="navigation-section-title">
                    {section.section}
                  </div>

                  {section.items.map((item) => (
                    <NavLink
                      key={item.id}
                      to={item.path}
                      end={item.path === "/"}
                      className={({ isActive }) =>
                        ["navigation-item", isActive ? "active" : ""]
                          .filter(Boolean)
                          .join(" ")
                      }
                      onClick={() => setMobileMenuOpen(false)}
                    >
                      <span className="navigation-item-icon" aria-hidden="true">
                        {getNavigationIcon(item.id)}
                      </span>

                      <span>{item.label}</span>
                    </NavLink>
                  ))}
                </div>
              ))}
            </nav>

            <div className="sidebar-status">
              <span className="status-indicator" aria-hidden="true" />

              <div>
                <div className="sidebar-status-title">System operational</div>

                <div className="sidebar-status-subtitle">
                  All core services available
                </div>
              </div>
            </div>
          </aside>
        </div>
      )}

      {/* ============================================================
          Main Application
          ============================================================ */}

      <div className="app-main">
        <header className="app-topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="mobile-menu-button"
              onClick={() => setMobileMenuOpen(true)}
              aria-label="Open navigation"
              title="Open navigation"
            >
              ☰
            </button>

            <div className="topbar-context">
              <span className="topbar-context-label">Workspace</span>

              <span className="topbar-context-divider">/</span>

              <span className="topbar-context-current">
                {currentNavigationItem.label}
              </span>
            </div>
          </div>

          <div className="topbar-actions">
            <button
              type="button"
              className="topbar-icon-button"
              aria-label="Search"
              title="Search"
            >
              ⌕
            </button>

            <button
              type="button"
              className="topbar-icon-button"
              aria-label="Notifications"
              title="Notifications"
            >
              ♢
            </button>

            <button
              type="button"
              className="theme-toggle-button"
              onClick={toggleTheme}
              aria-label={
                theme === "light"
                  ? "Switch to dark mode"
                  : "Switch to light mode"
              }
              title={
                theme === "light"
                  ? "Switch to dark mode"
                  : "Switch to light mode"
              }
            >
              <span aria-hidden="true">{theme === "light" ? "☾" : "☀"}</span>

              <span className="theme-toggle-label">
                {theme === "light" ? "Dark" : "Light"}
              </span>
            </button>

            <div className="topbar-user">
              <div className="topbar-avatar">U</div>

              <div className="topbar-user-info">
                <span className="topbar-user-name">Operator</span>

                <span className="topbar-user-role">Workspace user</span>
              </div>
            </div>
          </div>
        </header>

        <main className="app-content">{children}</main>
      </div>
    </div>
  );
}

/**
 * Temporary semantic icons.
 *
 * These can later be replaced by the project's
 * icon system without changing navigation data.
 */
function getNavigationIcon(id) {
  const icons = {
    dashboard: "◈",
    transform: "✦",
    executions: "▶",
    artifacts: "□",
    knowledge: "◇",
    workflows: "⌘",
    agents: "◎",
    integrations: "⛓",
    settings: "⚙",
  };

  return icons[id] ?? "•";
}

export default AppShell;
