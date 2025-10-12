document.addEventListener('DOMContentLoaded', function () {
  // Find all question blocks on the page
  var questionElements = Array.from(document.querySelectorAll('.question'));
  var submitButton = document.getElementById('submit-quiz');
  var resultElement = document.getElementById('result');
  var restartButton = document.getElementById('restart-quiz');

  // Helper: set up each question so only one option can be selected
  function setupSingleSelect() {
    questionElements.forEach(function (q) {
      var optionButtons = Array.from(q.querySelectorAll('.options button'));
      optionButtons.forEach(function (button) {
        button.addEventListener('click', function () {
          // Remove 'selected' from other buttons in this question
          optionButtons.forEach(function (b) { b.classList.remove('selected'); });
          // Add 'selected' to the clicked button
          button.classList.add('selected');
        });
      });
    });
  }

  // Helper: grade the quiz when user clicks Submit
  function gradeQuiz() {
    var total = questionElements.length;
    var correctCount = 0;

    questionElements.forEach(function (q) {
      var selected = q.querySelector('.options button.selected');

      if (!selected) {
        // No selection made: show the correct answer only
        q.classList.add('unanswered');
        var correctBtn = q.querySelector('.options button[data-correct="true"]');
        if (correctBtn) correctBtn.classList.add('correct');
        return;
      }

      // dataset.correct is a string, compare to 'true'
      var isCorrect = selected.dataset.correct === 'true';
      if (isCorrect) {
        correctCount += 1;
        selected.classList.add('correct');
      } else {
        selected.classList.add('wrong');
        var correctBtn = q.querySelector('.options button[data-correct="true"]');
        if (correctBtn) correctBtn.classList.add('correct');
      }
    });

    // Show the score
    resultElement.textContent = 'You scored ' + correctCount + ' out of ' + total;

    // Disable further changes after grading
    questionElements.forEach(function (q) {
      q.querySelectorAll('.options button').forEach(function (b) { b.disabled = true; });
    });
    submitButton.disabled = true;
  }

  // Helper: reset the quiz to initial state
  function resetQuiz() {
    resultElement.textContent = '';
    submitButton.disabled = false;

    questionElements.forEach(function (q) {
      q.classList.remove('unanswered');
      q.querySelectorAll('.options button').forEach(function (b) {
        b.disabled = false;
        b.classList.remove('selected');
        b.classList.remove('correct');
        b.classList.remove('wrong');
      });
    });
  }

  // Set up event listeners
  setupSingleSelect();
  submitButton.addEventListener('click', gradeQuiz);
  restartButton.addEventListener('click', resetQuiz);
});