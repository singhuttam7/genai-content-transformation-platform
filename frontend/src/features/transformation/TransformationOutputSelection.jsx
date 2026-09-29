import { transformationTypes } from "./constants";

function TransformationOutputSelection({ requestedOutputs, onChange }) {
  function toggleOutput(value) {
    if (requestedOutputs.includes(value)) {
      onChange(requestedOutputs.filter((output) => output !== value));
      return;
    }

    onChange([...requestedOutputs, value]);
  }

  return (
    <section className="transformation-card">
      <div className="transformation-card-header">
        <div>
          <span className="transformation-card-kicker">STEP 4</span>

          <h2>Select output artifacts</h2>

          <p>
            Choose one or more communication artifacts to generate from the same
            source.
          </p>
        </div>
      </div>

      <div className="transformation-type-grid">
        {transformationTypes.map((type) => {
          const selected = requestedOutputs.includes(type.value);

          return (
            <button
              key={type.value}
              type="button"
              className={`transformation-type-option ${
                selected ? "selected" : ""
              }`}
              onClick={() => toggleOutput(type.value)}
              aria-pressed={selected}
            >
              <span className="transformation-type-label">{type.label}</span>

              <span className="transformation-type-description">
                {type.description}
              </span>

              <span>{selected ? "Selected" : "Select"}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

export default TransformationOutputSelection;
