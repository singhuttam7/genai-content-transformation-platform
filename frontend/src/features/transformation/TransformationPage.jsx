import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { useWorkspace } from "../../context/WorkspaceContext";
import { createExecution } from "../../services/api/executions";
import { getDevelopmentWorkflow } from "../../services/api/developmentWorkflow";
import { createTransformation } from "../../services/api/transformations";
import { createTransformationSource } from "./createTransformationSource";
import { sourceTypes, transformationTypes } from "./constants";
import { buildTransformationPayload } from "./transformationPayload";

import TransformationConfiguration from "./TransformationConfiguration";
import TransformationOutputSelection from "./TransformationOutputSelection";

function getResponseData(response) {
  return response?.data ?? response;
}

function getErrorMessage(error) {
  return (
    error?.response?.data?.error?.message ||
    error?.response?.data?.detail ||
    error?.message ||
    "An unexpected error occurred."
  );
}

function TransformationPage() {
  const navigate = useNavigate();

  const {
    workspace,
    loading: workspaceLoading,
    error: workspaceError,
  } = useWorkspace();

  const [createdSource, setCreatedSource] = useState(null);

  const [sourceCreating, setSourceCreating] = useState(false);

  const [sourceError, setSourceError] = useState(null);

  const [createdTransformation, setCreatedTransformation] = useState(null);

  const [transformationCreating, setTransformationCreating] = useState(false);

  const [transformationError, setTransformationError] = useState(null);

  const [createdExecution, setCreatedExecution] = useState(null);

  const [executionCreating, setExecutionCreating] = useState(false);

  const [executionError, setExecutionError] = useState(null);

  const [sourceType, setSourceType] = useState("text");

  const [transformationType, setTransformationType] =
    useState("executive_summary");

  const [sourceTitle, setSourceTitle] = useState("");

  const [sourceContent, setSourceContent] = useState("");

  const [sourceUrl, setSourceUrl] = useState("");

  const [configuration, setConfiguration] = useState({
    objective: "",
    audience: "",
    tone: "",
    language: "",
    detailLevel: "",
    style: "",
  });

  const [requestedOutputs, setRequestedOutputs] = useState([
    "executive_summary",
  ]);

  function resetDownstreamState() {
    setCreatedTransformation(null);
    setTransformationError(null);
    setCreatedExecution(null);
    setExecutionError(null);
  }

  function resetSourceState() {
    setCreatedSource(null);
    setSourceError(null);
    resetDownstreamState();
  }

  async function handleCreateSource() {
    setSourceError(null);
    resetDownstreamState();

    if (workspaceLoading) {
      return;
    }

    if (workspaceError || !workspace?.project?.id) {
      setSourceError("Workspace is unavailable. Please try again.");
      return;
    }

    if (!sourceTitle.trim()) {
      setSourceError("Please enter a source title.");
      return;
    }

    if (sourceType === "url" && !sourceUrl.trim()) {
      setSourceError("Please enter a source URL.");
      return;
    }

    if (sourceType !== "url" && !sourceContent.trim()) {
      setSourceError("Please provide source content.");
      return;
    }

    try {
      setSourceCreating(true);

      const response = await createTransformationSource({
        sourceType,
        title: sourceTitle.trim(),
        content: sourceContent.trim(),
        url: sourceUrl.trim(),
        projectId: workspace.project.id,
      });

      const source = getResponseData(response);

      setCreatedSource(source);
    } catch (error) {
      setSourceError(getErrorMessage(error));
    } finally {
      setSourceCreating(false);
    }
  }

  async function handleCreateTransformation() {
    setTransformationError(null);
    setCreatedTransformation(null);
    setExecutionError(null);
    setCreatedExecution(null);

    if (workspaceLoading) {
      return;
    }

    if (workspaceError || !workspace?.project?.id) {
      setTransformationError("Workspace is unavailable. Please try again.");
      return;
    }

    if (!createdSource?.source_id) {
      setTransformationError(
        "Create a source before creating the transformation.",
      );
      return;
    }

    if (!configuration.language.trim()) {
      setTransformationError("Please select a language.");
      return;
    }

    if (requestedOutputs.length === 0) {
      setTransformationError("Select at least one output artifact.");
      return;
    }

    try {
      setTransformationCreating(true);

      const payload = buildTransformationPayload({
        projectId: workspace.project.id,
        sourceId: createdSource.source_id,
        transformationType,
        configuration,
        requestedOutputs,
      });

      const response = await createTransformation(payload);

      const transformation = getResponseData(response);

      setCreatedTransformation(transformation);
    } catch (error) {
      setTransformationError(getErrorMessage(error));
    } finally {
      setTransformationCreating(false);
    }
  }

  async function handleCreateExecution() {
    setExecutionError(null);
    setCreatedExecution(null);

    if (!createdTransformation?.id) {
      setExecutionError(
        "Create a transformation before starting an execution.",
      );
      return;
    }

    if (!createdSource?.source_id) {
      setExecutionError(
        "Source information is unavailable for this execution.",
      );
      return;
    }

    if (!workspace?.project?.id) {
      setExecutionError("Workspace is unavailable. Please try again.");
      return;
    }

    try {
      setExecutionCreating(true);

      const workflowResponse = await getDevelopmentWorkflow({
        projectId: workspace.project.id,
        transformationType,
      });

      const workflow = getResponseData(workflowResponse);

      if (!workflow?.id) {
        throw new Error("No workflow was returned for this transformation.");
      }

      const executionResponse = await createExecution({
        transformationId: createdTransformation.id,

        workflowId: workflow.id,

        executionContext: {
          source_id: createdSource.source_id,

          transformation_type: transformationType,
        },
      });

      const execution = getResponseData(executionResponse);

      setCreatedExecution(execution);
    } catch (error) {
      setExecutionError(getErrorMessage(error));
    } finally {
      setExecutionCreating(false);
    }
  }

  function handleSourceTypeChange(value) {
    setSourceType(value);
    resetSourceState();
  }

  function handleTransformationTypeChange(value) {
    setTransformationType(value);
    resetDownstreamState();
  }

  function handleConfigurationChange(nextConfiguration) {
    setConfiguration(nextConfiguration);
    resetDownstreamState();
  }

  function handleOutputChange(nextOutputs) {
    setRequestedOutputs(nextOutputs);
    resetDownstreamState();
  }

  const executionCompleted = createdExecution?.status === "COMPLETED";

  return (
    <section className="transformation-page">
      <div className="transformation-page-header">
        <div>
          <span className="eyebrow">CONTENT TRANSFORMATION</span>

          <h1>Transform content.</h1>

          <p>
            Configure a source and transformation intent, then generate
            audience-ready communication artifacts from the same content.
          </p>
        </div>

        <div className="transformation-step-indicator">
          <span className="transformation-step active">1</span>

          <span className="transformation-step-line" />

          <span
            className={`transformation-step ${createdSource ? "active" : ""}`}
          >
            2
          </span>

          <span className="transformation-step-line" />

          <span
            className={`transformation-step ${
              createdTransformation ? "active" : ""
            }`}
          >
            3
          </span>
        </div>
      </div>

      <div className="transformation-layout">
        <div className="transformation-main">
          <section className="transformation-card">
            <div className="transformation-card-header">
              <div>
                <span className="transformation-card-kicker">STEP 1</span>

                <h2>Choose your source</h2>

                <p>
                  Provide the content that the platform should analyze and
                  transform.
                </p>
              </div>
            </div>

            <div className="source-type-grid">
              {sourceTypes.map((type) => (
                <button
                  key={type.value}
                  type="button"
                  className={`source-type-option ${
                    sourceType === type.value ? "selected" : ""
                  }`}
                  onClick={() => handleSourceTypeChange(type.value)}
                >
                  <span className="source-type-label">{type.label}</span>

                  <span className="source-type-description">
                    {type.description}
                  </span>
                </button>
              ))}
            </div>

            <div className="transformation-form-grid">
              <label className="form-field">
                <span>Source title</span>

                <input
                  type="text"
                  value={sourceTitle}
                  onChange={(event) => {
                    setSourceTitle(event.target.value);
                    setCreatedSource(null);
                    setSourceError(null);
                    resetDownstreamState();
                  }}
                  placeholder="Enter a descriptive title"
                />
              </label>

              {sourceType === "url" ? (
                <label className="form-field">
                  <span>Source URL</span>

                  <input
                    type="url"
                    value={sourceUrl}
                    onChange={(event) => {
                      setSourceUrl(event.target.value);
                      setCreatedSource(null);
                      setSourceError(null);
                      resetDownstreamState();
                    }}
                    placeholder="https://example.com/article"
                  />
                </label>
              ) : (
                <label className="form-field form-field-full">
                  <span>
                    {sourceType === "prompt" ? "Prompt" : "Source content"}
                  </span>

                  <textarea
                    value={sourceContent}
                    onChange={(event) => {
                      setSourceContent(event.target.value);
                      setCreatedSource(null);
                      setSourceError(null);
                      resetDownstreamState();
                    }}
                    placeholder={
                      sourceType === "prompt"
                        ? "Describe what the AI should transform..."
                        : "Paste the source content here..."
                    }
                    rows={10}
                  />
                </label>
              )}
            </div>
          </section>

          <section className="transformation-card">
            <div className="transformation-card-header">
              <div>
                <span className="transformation-card-kicker">STEP 2</span>

                <h2>Choose transformation</h2>

                <p>
                  Select the primary communication artifact you want to
                  generate.
                </p>
              </div>
            </div>

            <div className="transformation-type-grid">
              {transformationTypes.map((type) => (
                <button
                  key={type.value}
                  type="button"
                  className={`transformation-type-option ${
                    transformationType === type.value ? "selected" : ""
                  }`}
                  onClick={() => handleTransformationTypeChange(type.value)}
                >
                  <span className="transformation-type-label">
                    {type.label}
                  </span>

                  <span className="transformation-type-description">
                    {type.description}
                  </span>
                </button>
              ))}
            </div>
          </section>

          <TransformationConfiguration
            objective={configuration.objective}
            audience={configuration.audience}
            tone={configuration.tone}
            language={configuration.language}
            detailLevel={configuration.detailLevel}
            style={configuration.style}
            onChange={handleConfigurationChange}
          />

          <TransformationOutputSelection
            requestedOutputs={requestedOutputs}
            onChange={handleOutputChange}
          />
        </div>

        <aside className="transformation-sidebar">
          <section className="transformation-summary-card">
            <span className="transformation-card-kicker">TRANSFORMATION</span>

            <div className="summary-row">
              <span>Workspace</span>

              <strong>
                {workspaceLoading
                  ? "Loading..."
                  : workspaceError
                    ? "Unavailable"
                    : workspace?.project?.name || "Unavailable"}
              </strong>
            </div>

            <h2>
              {
                transformationTypes.find(
                  (type) => type.value === transformationType,
                )?.label
              }
            </h2>

            <div className="summary-divider" />

            <div className="summary-row">
              <span>Source</span>

              <strong>
                {sourceTypes.find((type) => type.value === sourceType)?.label}
              </strong>
            </div>

            <div className="summary-row">
              <span>Content</span>

              <strong>
                {sourceContent.trim() || sourceUrl.trim()
                  ? "Provided"
                  : "Not provided"}
              </strong>
            </div>

            <button
              type="button"
              className="primary-action-button"
              onClick={handleCreateSource}
              disabled={sourceCreating || workspaceLoading}
            >
              {sourceCreating ? "Creating source..." : "Create source"}
            </button>

            {createdSource && (
              <>
                <div className="summary-row">
                  <span>Source status</span>

                  <strong>Created</strong>
                </div>

                <div className="summary-row">
                  <span>Source ID</span>

                  <strong>{createdSource.source_id}</strong>
                </div>

                <button
                  type="button"
                  className="primary-action-button"
                  onClick={handleCreateTransformation}
                  disabled={transformationCreating || workspaceLoading}
                >
                  {transformationCreating
                    ? "Creating transformation..."
                    : "Create transformation"}
                </button>
              </>
            )}

            {sourceError && (
              <div className="summary-row">
                <span>Source error</span>

                <strong>{sourceError}</strong>
              </div>
            )}

            {createdTransformation && (
              <>
                <div className="summary-row">
                  <span>Transformation status</span>

                  <strong>{createdTransformation.status}</strong>
                </div>

                <div className="summary-row">
                  <span>Transformation ID</span>

                  <strong>{createdTransformation.id}</strong>
                </div>

                <button
                  type="button"
                  className="primary-action-button"
                  onClick={handleCreateExecution}
                  disabled={executionCreating}
                >
                  {executionCreating
                    ? "Starting execution..."
                    : "Start execution"}
                </button>
              </>
            )}

            {transformationError && (
              <div className="summary-row">
                <span>Transformation error</span>

                <strong>{transformationError}</strong>
              </div>
            )}

            {createdExecution && (
              <>
                <div className="summary-row">
                  <span>Execution status</span>

                  <strong>{createdExecution.status}</strong>
                </div>

                <div className="summary-row">
                  <span>Execution ID</span>

                  <strong>{createdExecution.id}</strong>
                </div>

                {executionCompleted && (
                  <button
                    type="button"
                    className="secondary-action-button"
                    onClick={() => navigate("/artifacts")}
                  >
                    View artifacts
                  </button>
                )}
              </>
            )}

            {executionError && (
              <div className="summary-row">
                <span>Execution error</span>

                <strong>{executionError}</strong>
              </div>
            )}
          </section>
        </aside>
      </div>
    </section>
  );
}

export default TransformationPage;
