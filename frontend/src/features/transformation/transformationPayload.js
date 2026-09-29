export function buildTransformationPayload({
  projectId,
  sourceId,
  transformationType,
  configuration,
  requestedOutputs,
}) {
  return {
    project_id: projectId,
    source_id: sourceId,
    transformation_type: transformationType,
    objective: configuration.objective,
    audience: configuration.audience,
    tone: configuration.tone,
    language: configuration.language,
    detail_level: configuration.detailLevel,
    style: configuration.style,
    requested_outputs: requestedOutputs,
    configuration: {},
  };
}
