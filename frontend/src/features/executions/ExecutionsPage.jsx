import { useMemo } from "react";

import ExecutionErrorState from "./components/ExecutionErrorState";
import ExecutionHeader from "./components/ExecutionHeader";
import ExecutionInspector from "./components/ExecutionInspector";
import ExecutionLibrary from "./components/ExecutionLibrary";
import ExecutionSummary from "./components/ExecutionSummary";
import { useExecutions } from "./hooks/useExecutions";

function getExecutionStatusCounts(executions) {
  return executions.reduce(
    (counts, execution) => {
      const status = String(execution?.status ?? "").toLowerCase();

      counts.total += 1;

      if (
        status === "completed" ||
        status === "success" ||
        status === "succeeded"
      ) {
        counts.completed += 1;
      } else if (status === "running" || status === "in_progress") {
        counts.running += 1;
      } else if (status === "failed" || status === "error") {
        counts.failed += 1;
      }

      return counts;
    },
    {
      total: 0,
      completed: 0,
      running: 0,
      failed: 0,
    },
  );
}

function ExecutionsPage() {
  const {
    executions,
    selectedExecution,
    loading,
    detailLoading,
    error,
    selectExecution,
    refresh,
  } = useExecutions();

  const statusCounts = useMemo(
    () => getExecutionStatusCounts(executions),
    [executions],
  );

  return (
    <section className="executions-page">
      <ExecutionHeader onRefresh={refresh} refreshing={loading} />

      {error && (
        <ExecutionErrorState
          message={error}
          onRetry={refresh}
          retrying={loading}
        />
      )}

      <ExecutionSummary
        total={statusCounts.total}
        completed={statusCounts.completed}
        running={statusCounts.running}
        failed={statusCounts.failed}
      />

      <div className="executions-layout">
        <ExecutionLibrary
          executions={executions}
          selectedExecutionId={selectedExecution?.id ?? null}
          loading={loading}
          onSelect={selectExecution}
        />

        <ExecutionInspector
          execution={selectedExecution}
          loading={detailLoading}
        />
      </div>
    </section>
  );
}

export default ExecutionsPage;
