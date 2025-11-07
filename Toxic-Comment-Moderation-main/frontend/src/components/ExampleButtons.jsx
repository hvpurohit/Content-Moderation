import React from 'react'

const EXAMPLES = [
  {
    label: 'Benign',
    text: 'Thank you for your help! Have a great day.'
  },
  {
    label: 'Insult',
    text: 'You are an idiot and your opinion is worthless.'
  },
  {
    label: 'Threat',
    text: 'I will kick you.'
  },
  {
    label: 'Profanity',
    text: 'This is complete garbage and you should be ashamed.'
  },
  {
    label: 'Neutral',
    text: 'I think we should consider all options before making a decision.'
  }
]

function ExampleButtons({ onExampleClick }) {
  return (
    <div className="example-buttons-container">
      <p className="example-label">Try examples:</p>
      <div className="example-buttons">
        {EXAMPLES.map((example, index) => (
          <button
            key={index}
            className="btn btn-example"
            onClick={() => onExampleClick(example.text)}
            type="button"
          >
            {example.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export default ExampleButtons

