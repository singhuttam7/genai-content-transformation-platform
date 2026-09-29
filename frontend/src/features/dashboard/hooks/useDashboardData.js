import { useCallback, useEffect, useState } from "react";

import { getDashboardData } from "../services/dashboardService";

const INITIAL_STATE = {
  data: null,
  loading: true,
  refreshing: false,
  error: null,
};

export function useDashboardData() {
  const [state, setState] = useState(INITIAL_STATE);

  const loadDashboard = useCallback(async (isRefresh = false) => {
    setState((current) => ({
      ...current,
      loading: !isRefresh && current.data === null,
      refreshing: isRefresh,
      error: null,
    }));

    try {
      const dashboardData = await getDashboardData();

      setState({
        data: dashboardData,
        loading: false,
        refreshing: false,
        error: null,
      });
    } catch (error) {
      setState((current) => ({
        ...current,
        loading: false,
        refreshing: false,
        error: error?.message || "Unable to load dashboard data.",
      }));
    }
  }, []);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  const refresh = useCallback(() => {
    return loadDashboard(true);
  }, [loadDashboard]);

  return {
    data: state.data,
    loading: state.loading,
    refreshing: state.refreshing,
    error: state.error,
    refresh,
  };
}
