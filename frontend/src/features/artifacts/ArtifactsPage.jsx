import { useMemo } from "react";

import ArtifactErrorState from "./components/ArtifactErrorState";
import ArtifactHeader from "./components/ArtifactHeader";
import ArtifactInspector from "./components/ArtifactInspector";
import ArtifactLibrary from "./components/ArtifactLibrary";
import ArtifactSummary from "./components/ArtifactSummary";
import { useArtifacts } from "./hooks/useArtifacts";

function getArtifactStatusCounts(artifacts) {
  return artifacts.reduce(
    (counts, artifact) => {
      const status = String(artifact?.status ?? "").toLowerCase();

      if (
        status === "completed" ||
        status === "success" ||
        status === "succeeded" ||
        status === "generated"
      ) {
        counts.completed += 1;
      }

      if (status === "failed" || status === "error") {
        counts.failed += 1;
      }

      return counts;
    },
    {
      completed: 0,
      failed: 0,
    },
  );
}

function ArtifactsPage() {
  const {
    artifacts,
    selectedArtifact,
    loading,
    detailLoading,
    error,
    selectArtifact,
    refresh,
  } = useArtifacts();

  const statusCounts = useMemo(
    () => getArtifactStatusCounts(artifacts),
    [artifacts],
  );

  return (
    <section className="artifacts-page">
      <ArtifactHeader onRefresh={refresh} refreshing={loading} />

      {error && (
        <ArtifactErrorState
          message={error}
          onRetry={refresh}
          retrying={loading}
        />
      )}

      <ArtifactSummary
        total={artifacts.length}
        completed={statusCounts.completed}
        failed={statusCounts.failed}
        selected={Boolean(selectedArtifact)}
      />

      <div className="artifacts-layout">
        <ArtifactLibrary
          artifacts={artifacts}
          selectedArtifactId={selectedArtifact?.id ?? null}
          loading={loading}
          onSelect={selectArtifact}
        />

        <ArtifactInspector
          artifact={selectedArtifact}
          loading={detailLoading}
        />
      </div>
    </section>
  );
}

export default ArtifactsPage;
