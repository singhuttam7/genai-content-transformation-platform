import { useCallback, useEffect, useState } from "react";

import { useWorkspace } from "../../../context/WorkspaceContext";

import { fetchKnowledgeSources } from "../services/knowledgeService";

const INITIAL_STATE = {
  sources: [],
  selectedSource: null,
  loading: true,
  detailLoading: false,
  error: null,
};

export function useKnowledge() {
  const {
    workspace,
    loading: workspaceLoading,
    error: workspaceError,
  } = useWorkspace();

  const [state, setState] = useState(INITIAL_STATE);

  const projectId = workspace?.project?.id ?? null;

  const loadKnowledge = useCallback(
    async (isRefresh = false) => {
      if (workspaceLoading) {
        return;
      }

      if (workspaceError) {
        setState((current) => ({
          ...current,
          loading: false,
          error: workspaceError,
        }));

        return;
      }

      if (!projectId) {
        setState((current) => ({
          ...current,
          loading: false,
          error: "Workspace project is unavailable.",
        }));

        return;
      }

      setState((current) => ({
        ...current,
        loading: !isRefresh && current.sources.length === 0,
        error: null,
      }));

      try {
        const result = await fetchKnowledgeSources({
          projectId,
        });

        setState((current) => {
          const selectedSource = current.selectedSource;

          const selectedStillExists =
            selectedSource &&
            result.items.some((source) => source.id === selectedSource.id);

          return {
            ...current,
            sources: result.items,
            selectedSource: selectedStillExists ? selectedSource : null,
            loading: false,
            error: null,
          };
        });
      } catch (error) {
        setState((current) => ({
          ...current,
          loading: false,
          error: error?.message || "Unable to load knowledge sources.",
        }));
      }
    },
    [projectId, workspaceLoading, workspaceError],
  );

  const selectSource = useCallback((sourceId) => {
    if (!sourceId) {
      return;
    }

    setState((current) => {
      const source =
        current.sources.find((item) => item.id === sourceId) ?? null;

      return {
        ...current,
        selectedSource: source,
        detailLoading: false,
      };
    });
  }, []);

  const clearSelection = useCallback(() => {
    setState((current) => ({
      ...current,
      selectedSource: null,
      detailLoading: false,
    }));
  }, []);

  const refresh = useCallback(() => {
    return loadKnowledge(true);
  }, [loadKnowledge]);

  useEffect(() => {
    loadKnowledge();
  }, [loadKnowledge]);

  return {
    sources: state.sources,
    selectedSource: state.selectedSource,
    loading: state.loading || workspaceLoading,
    detailLoading: state.detailLoading,
    error: state.error || workspaceError || null,
    loadKnowledge,
    selectSource,
    clearSelection,
    refresh,
  };
}
