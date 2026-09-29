import { useCallback, useEffect, useState } from "react";

import { fetchArtifact, fetchArtifacts } from "../services/artifactService";

const INITIAL_STATE = {
  artifacts: [],
  selectedArtifact: null,
  loading: true,
  detailLoading: false,
  error: null,
};

export function useArtifacts() {
  const [state, setState] = useState(INITIAL_STATE);

  const loadArtifacts = useCallback(async (isRefresh = false) => {
    setState((current) => ({
      ...current,
      loading: !isRefresh && current.artifacts.length === 0,
      error: null,
    }));

    try {
      const result = await fetchArtifacts();

      setState((current) => {
        const selectedArtifact = current.selectedArtifact;

        const selectedStillExists =
          selectedArtifact &&
          result.items.some((artifact) => artifact.id === selectedArtifact.id);

        return {
          ...current,
          artifacts: result.items,
          selectedArtifact: selectedStillExists ? selectedArtifact : null,
          loading: false,
          error: null,
        };
      });
    } catch (error) {
      setState((current) => ({
        ...current,
        loading: false,
        error: error?.message || "Unable to load artifacts.",
      }));
    }
  }, []);

  const selectArtifact = useCallback(async (artifactId) => {
    if (!artifactId) {
      return;
    }

    setState((current) => ({
      ...current,
      detailLoading: true,
      error: null,
    }));

    try {
      const artifact = await fetchArtifact(artifactId);

      setState((current) => ({
        ...current,
        selectedArtifact: artifact,
        detailLoading: false,
        error: null,
      }));
    } catch (error) {
      setState((current) => ({
        ...current,
        detailLoading: false,
        error: error?.message || "Unable to load artifact details.",
      }));
    }
  }, []);

  const refresh = useCallback(() => {
    return loadArtifacts(true);
  }, [loadArtifacts]);

  useEffect(() => {
    loadArtifacts();
  }, [loadArtifacts]);

  return {
    artifacts: state.artifacts,
    selectedArtifact: state.selectedArtifact,
    loading: state.loading,
    detailLoading: state.detailLoading,
    error: state.error,
    loadArtifacts,
    selectArtifact,
    refresh,
  };
}
