document.getElementById('ckdTestForm').addEventListener('submit', function (e) {
    e.preventDefault();

    // Simulating test result
    const age = document.getElementById('age').value;
    const bloodPressure = document.getElementById('bloodPressure').value;
    const creatinine = document.getElementById('creatinine').value;

    if (age && bloodPressure && creatinine) {
        alert(`Test Complete! Age: ${age}, BP: ${bloodPressure}, Creatinine: ${creatinine}`);
    } else {
        alert('Please fill in all fields.');
    }
});
