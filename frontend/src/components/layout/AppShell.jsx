import { useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { navigation } from "../../constants/navigation";

function AppShell({ children }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  const currentNavigationItem =
    navigation
      .flatMap((section) => section.items)
      .find((item) => item.path === location.pathname) ??
    navigation[0].items[0];

  return (
    <div className="app-shell">
      {/* ============================================================
          Desktop Sidebar
          ============================================================ */}
      <aside className="app-sidebar">
        <div className="app-brand">
          <div className="app-brand-mark">✦</div>

          <div>
            <div className="app-brand-name">GenAI Platform</div>
            <div className="app-brand-subtitle">Content Transformation</div>
          </div>
        </div>

        <nav className="app-navigation">
          {navigation.map((section) => (
            <div className="navigation-section" key={section.section}>
              <div className="navigation-section-title">{section.section}</div>

              {section.items.map((item) => (
                <NavLink
                  key={item.id}
                  to={item.path}
                  end={item.path === "/"}
                  className={({ isActive }) =>
                    `navigation-item ${isActive ? "active" : ""}`
                  }
                  title={item.description}
                >
                  <span className="navigation-item-icon">
                    {getNavigationIcon(item.id)}
                  </span>

                  <span>{item.label}</span>
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        <div className="sidebar-status">
          <span className="status-indicator" />

          <div>
            <div className="sidebar-status-title">System operational</div>

            <div className="sidebar-status-subtitle">
              All core services available
            </div>
          </div>
        </div>
      </aside>

      {/* ============================================================
          Mobile Navigation
          ============================================================ */}
      {mobileMenuOpen && (
        <div
          className="mobile-navigation-overlay"
          onClick={() => setMobileMenuOpen(false)}
        >
          <aside
            className="mobile-sidebar"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="app-brand">
              <div className="app-brand-mark">✦</div>

              <div>
                <div className="app-brand-name">GenAI Platform</div>
                <div className="app-brand-subtitle">Content Transformation</div>
              </div>
            </div>

            <nav className="app-navigation">
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
                        `navigation-item ${isActive ? "active" : ""}`
                      }
                      onClick={() => setMobileMenuOpen(false)}
                    >
                      <span className="navigation-item-icon">
                        {getNavigationIcon(item.id)}
                      </span>

                      <span>{item.label}</span>
                    </NavLink>
                  ))}
                </div>
              ))}
            </nav>
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
            >
              ⌕
            </button>

            <button
              type="button"
              className="topbar-icon-button"
              aria-label="Notifications"
            >
              ♢
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
