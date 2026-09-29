import { useNavigate } from "react-router-dom";

function formatDate(value) {
  if (!value) {
    return "Unknown time";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "Unknown time";
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function formatArtifactType(value) {
  if (!value) {
    return "Generated artifact";
  }

  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function getStatusClass(status) {
  const normalized = String(status ?? "UNKNOWN").toLowerCase();

  return `execution-status-${normalized}`;
}

function RecentArtifacts({ artifacts = [] }) {
  const navigate = useNavigate();

  function openArtifacts() {
    navigate("/artifacts");
  }

  return (
    <section className="dashboard-recent-card">
      <div className="dashboard-section-heading dashboard-section-heading-with-action">
        <div>
          <span className="dashboard-section-kicker">Generated content</span>

          <h2>Recent artifacts</h2>
        </div>

        <button
          type="button"
          className="dashboard-section-link"
          onClick={openArtifacts}
        >
          View all
          <span aria-hidden="true">→</span>
        </button>
      </div>

      {artifacts.length === 0 ? (
        <div className="dashboard-empty-state">
          <div className="dashboard-empty-icon" aria-hidden="true">
            □
          </div>

          <strong>No artifacts yet</strong>

          <p>
            Generated content will appear here after a transformation completes.
          </p>

          <button
            type="button"
            className="secondary-action-button"
            onClick={() => navigate("/transform")}
          >
            Create transformation
          </button>
        </div>
      ) : (
        <div className="dashboard-artifact-list">
          {artifacts.map((artifact) => {
            const status = String(artifact?.status ?? "UNKNOWN").toUpperCase();

            return (
              <button
                key={artifact.id}
                type="button"
                className="dashboard-artifact-item"
                onClick={openArtifacts}
              >
                <span className="dashboard-artifact-icon" aria-hidden="true">
                  {getArtifactIcon(artifact?.artifact_type)}
                </span>

                <span className="dashboard-artifact-main">
                  <strong>
                    {artifact?.title ||
                      formatArtifactType(artifact?.artifact_type)}
                  </strong>

                  <span>{formatArtifactType(artifact?.artifact_type)}</span>
                </span>

                <span className="dashboard-artifact-time">
                  {formatDate(artifact?.created_at)}
                </span>

                <span
                  className={[
                    "execution-status",
                    getStatusClass(artifact?.status),
                  ].join(" ")}
                >
                  {status}
                </span>

                <span className="dashboard-artifact-arrow" aria-hidden="true">
                  →
                </span>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}

function getArtifactIcon(artifactType) {
  const icons = {
    advisory: "A",
    executive_summary: "E",
    social_media: "S",
    infographic: "I",
    presentation: "P",
    video: "V",
  };

  return icons[artifactType] ?? "□";
}

export default RecentArtifacts;
