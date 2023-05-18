var principal, rate, time, calculatedInterest;

function generateQuestion() {
    principal = getRandomNumber(1000, 10000);
    rate = getRandomNumber(1, 10);
    time = getRandomNumber(1, 5);
    calculatedInterest = principal * (Math.pow(1 + (rate / 100), time)) - principal;
    document.getElementById("question").innerHTML = "Calculate the compound interest for a principal amount of $" + principal + " at an interest rate of " + rate + "% per year for " + time + " years:";
}

function checkAnswer() {
    var userAnswer = parseFloat(document.getElementById("answerInput").value);
    var feedback = document.getElementById("feedback");

    if (Math.abs(userAnswer - calculatedInterest) < 0.01) {
        feedback.innerHTML = "Correct! The calculated compound interest is $" + calculatedInterest.toFixed(2);
    } else {
        feedback.innerHTML = "Incorrect. The correct answer is $" + calculatedInterest.toFixed(2);
    }

    generateQuestion();
}

function getRandomNumber(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Generate the first question when the page loads
window.onload = generateQuestion;