import { useNavigate } from "react-router-dom";

import KnowledgeHeader from "./components/KnowledgeHeader";
import KnowledgeSummary from "./components/KnowledgeSummary";
import KnowledgeLibrary from "./components/KnowledgeLibrary";
import KnowledgeListItem from "./components/KnowledgeListItem";
import KnowledgeInspector from "./components/KnowledgeInspector";
import KnowledgeMetadata from "./components/KnowledgeMetadata";
import KnowledgeChunkViewer from "./components/KnowledgeChunkViewer";
import KnowledgeEmptyState from "./components/KnowledgeEmptyState";
import KnowledgeErrorState from "./components/KnowledgeErrorState";
import { useKnowledge } from "./hooks/useKnowledge";

function KnowledgePage() {
  const navigate = useNavigate();

  const {
    sources,
    selectedSource,
    loading,
    error,
    refresh,
    selectSource,
    clearSelection,
  } = useKnowledge();

  const handleCreateSource = () => {
    navigate("/transform");
  };

  const handleSelectSource = (sourceId) => {
    selectSource(sourceId);
  };

  if (error && sources.length === 0) {
    return (
      <main className="knowledge-page">
        <KnowledgeHeader onRefresh={refresh} refreshing={loading} />

        <KnowledgeErrorState message={error} onRetry={refresh} />
      </main>
    );
  }

  if (!loading && sources.length === 0) {
    return (
      <main className="knowledge-page">
        <KnowledgeHeader onRefresh={refresh} refreshing={loading} />

        <KnowledgeEmptyState onCreateSource={handleCreateSource} />
      </main>
    );
  }

  return (
    <main className="knowledge-page">
      <KnowledgeHeader onRefresh={refresh} refreshing={loading} />

      <KnowledgeSummary sources={sources} />

      <div className="knowledge-workspace">
        <section className="knowledge-library-panel">
          <KnowledgeLibrary
            sources={sources}
            selectedSourceId={selectedSource?.id || null}
            loading={loading}
            onSelect={handleSelectSource}
          />

          {sources.length > 0 && (
            <div className="knowledge-list-items">
              {sources.map((source) => (
                <KnowledgeListItem
                  key={source.id}
                  source={source}
                  selected={source.id === selectedSource?.id}
                  onSelect={handleSelectSource}
                />
              ))}
            </div>
          )}
        </section>

        <section className="knowledge-detail-panel">
          <KnowledgeInspector
            source={selectedSource}
            onClose={clearSelection}
          />

          {selectedSource && (
            <>
              <KnowledgeMetadata source={selectedSource} />

              <KnowledgeChunkViewer source={selectedSource} />
            </>
          )}
        </section>
      </div>
    </main>
  );
}

export default KnowledgePage;
