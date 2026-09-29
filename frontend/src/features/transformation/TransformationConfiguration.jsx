import {
  commonAudiences,
  commonLanguages,
  commonStyles,
  commonTones,
  detailLevels,
} from "./constants";

function TransformationConfiguration({
  objective,
  audience,
  tone,
  language,
  detailLevel,
  style,
  onChange,
}) {
  function updateField(field, value) {
    onChange({
      objective,
      audience,
      tone,
      language,
      detailLevel,
      style,
      [field]: value,
    });
  }

  return (
    <section className="transformation-card">
      <div className="transformation-card-header">
        <div>
          <span className="transformation-card-kicker">STEP 3</span>

          <h2>Configure the transformation</h2>

          <p>
            Define how the generated content should communicate with its
            intended audience.
          </p>
        </div>
      </div>

      <div className="transformation-form-grid">
        <label className="form-field form-field-full">
          <span>Communication objective</span>

          <textarea
            value={objective}
            onChange={(event) => updateField("objective", event.target.value)}
            placeholder="What should this transformation achieve?"
            rows={4}
          />
        </label>

        <label className="form-field">
          <span>Target audience</span>

          <select
            value={audience}
            onChange={(event) => updateField("audience", event.target.value)}
          >
            <option value="">Select an audience</option>

            {commonAudiences.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <label className="form-field">
          <span>Tone</span>

          <select
            value={tone}
            onChange={(event) => updateField("tone", event.target.value)}
          >
            <option value="">Select a tone</option>

            {commonTones.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <label className="form-field">
          <span>Language</span>

          <select
            value={language}
            onChange={(event) => updateField("language", event.target.value)}
          >
            <option value="">Select a language</option>

            {commonLanguages.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>

        <label className="form-field">
          <span>Detail level</span>

          <select
            value={detailLevel}
            onChange={(event) => updateField("detailLevel", event.target.value)}
          >
            <option value="">Select detail level</option>

            {detailLevels.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <label className="form-field form-field-full">
          <span>Content style</span>

          <select
            value={style}
            onChange={(event) => updateField("style", event.target.value)}
          >
            <option value="">Select a style</option>

            {commonStyles.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>
    </section>
  );
}

export default TransformationConfiguration;
