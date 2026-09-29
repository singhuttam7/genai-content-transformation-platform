import { useCallback, useEffect, useState } from "react";

import { fetchKnowledgeSources } from "../services/knowledgeService";

const INITIAL_STATE = {
  sources: [],
  selectedSource: null,
  loading: true,
  detailLoading: false,
  error: null,
};

export function useKnowledge() {
  const [state, setState] = useState(INITIAL_STATE);

  const loadKnowledge = useCallback(async (isRefresh = false) => {
    setState((current) => ({
      ...current,
      loading: !isRefresh && current.sources.length === 0,
      error: null,
    }));

    try {
      const result = await fetchKnowledgeSources();

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
  }, []);

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
    loading: state.loading,
    detailLoading: state.detailLoading,
    error: state.error,
    loadKnowledge,
    selectSource,
    clearSelection,
    refresh,
  };
}
