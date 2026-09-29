import { createContext, useContext, useEffect, useState } from "react";
import { getDevelopmentWorkspace } from "../services/api/developmentWorkspace";

const WorkspaceContext = createContext(null);

export function WorkspaceProvider({ children }) {
  const [workspace, setWorkspace] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadWorkspace() {
      try {
        setLoading(true);
        setError(null);

        const data = await getDevelopmentWorkspace();

        if (!cancelled) {
          setWorkspace(data);
        }
      } catch (workspaceError) {
        if (!cancelled) {
          setError(workspaceError);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadWorkspace();

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <WorkspaceContext.Provider
      value={{
        workspace,
        loading,
        error,
      }}
    >
      {children}
    </WorkspaceContext.Provider>
  );
}

export function useWorkspace() {
  const context = useContext(WorkspaceContext);

  if (!context) {
    throw new Error("useWorkspace must be used inside a WorkspaceProvider.");
  }

  return context;
}
