import { useCallback, useEffect, useState } from "react";

import { fetchExecution, fetchExecutions } from "../services/executionService";

const INITIAL_STATE = {
  executions: [],
  selectedExecution: null,
  loading: true,
  detailLoading: false,
  error: null,
};

export function useExecutions() {
  const [state, setState] = useState(INITIAL_STATE);

  const loadExecutions = useCallback(async (isRefresh = false) => {
    setState((current) => ({
      ...current,
      loading: !isRefresh && current.executions.length === 0,
      error: null,
    }));

    try {
      const result = await fetchExecutions();

      setState((current) => {
        const selectedExecution = current.selectedExecution;

        const selectedStillExists =
          selectedExecution &&
          result.items.some(
            (execution) => execution.id === selectedExecution.id,
          );

        return {
          ...current,
          executions: result.items,
          selectedExecution: selectedStillExists ? selectedExecution : null,
          loading: false,
          error: null,
        };
      });
    } catch (error) {
      setState((current) => ({
        ...current,
        loading: false,
        error: error?.message || "Unable to load executions.",
      }));
    }
  }, []);

  const selectExecution = useCallback(async (executionId) => {
    if (!executionId) {
      return;
    }

    setState((current) => ({
      ...current,
      detailLoading: true,
      error: null,
    }));

    try {
      const execution = await fetchExecution(executionId);

      setState((current) => ({
        ...current,
        selectedExecution: execution,
        detailLoading: false,
        error: null,
      }));
    } catch (error) {
      setState((current) => ({
        ...current,
        detailLoading: false,
        error: error?.message || "Unable to load execution details.",
      }));
    }
  }, []);

  const refresh = useCallback(() => {
    return loadExecutions(true);
  }, [loadExecutions]);

  useEffect(() => {
    loadExecutions();
  }, [loadExecutions]);

  return {
    executions: state.executions,
    selectedExecution: state.selectedExecution,
    loading: state.loading,
    detailLoading: state.detailLoading,
    error: state.error,
    loadExecutions,
    selectExecution,
    refresh,
  };
}
