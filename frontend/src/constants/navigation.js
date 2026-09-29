/**
 * Central navigation configuration for the
 * GenAI Content Transformation Platform.
 *
 * Navigation is kept separate from UI components so
 * future platform modules can be added without changing
 * the application shell implementation.
 */

export const navigation = [
  {
    section: "Workspace",
    items: [
      {
        id: "dashboard",
        label: "Dashboard",
        path: "/",
        description: "Overview of the content transformation workspace.",
      },
      {
        id: "transform",
        label: "Transform",
        path: "/transform",
        description: "Create a new content transformation.",
      },
      {
        id: "executions",
        label: "Executions",
        path: "/executions",
        description: "Monitor transformation executions and processing status.",
      },
      {
        id: "artifacts",
        label: "Artifacts",
        path: "/artifacts",
        description: "Review and manage generated communication artifacts.",
      },
    ],
  },

  {
    section: "Intelligence",
    items: [
      {
        id: "knowledge",
        label: "Knowledge",
        path: "/knowledge",
        description: "Manage knowledge sources and contextual information.",
      },
      {
        id: "workflows",
        label: "Workflows",
        path: "/workflows",
        description: "Build and manage automated transformation workflows.",
      },
      {
        id: "agents",
        label: "Agents",
        path: "/agents",
        description: "Manage specialized AI transformation agents.",
      },
    ],
  },

  {
    section: "Platform",
    items: [
      {
        id: "integrations",
        label: "Integrations",
        path: "/integrations",
        description: "Configure external services and platform integrations.",
      },
      {
        id: "settings",
        label: "Settings",
        path: "/settings",
        description: "Configure platform and workspace preferences.",
      },
    ],
  },
];

/**
 * Primary workspace navigation.
 */
export const primaryNavigation = navigation
  .flatMap((section) => section.items)
  .filter((item) =>
    ["dashboard", "transform", "executions", "artifacts"].includes(item.id),
  );

/**
 * AI and intelligence related navigation.
 */
export const intelligenceNavigation = navigation
  .flatMap((section) => section.items)
  .filter((item) =>
    ["knowledge", "workflows", "agents"].includes(item.id),
  );

/**
 * Platform configuration navigation.
 */
export const platformNavigation = navigation
  .flatMap((section) => section.items)
  .filter((item) =>
    ["integrations", "settings"].includes(item.id),
  );