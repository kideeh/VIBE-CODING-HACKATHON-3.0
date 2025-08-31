// AJAX for instant mood detection
const form = document.querySelector("#journalForm");
const result = document.querySelector("#moodResult");

if (form) {
    form.addEventListener("submit", async e => {
        e.preventDefault();
        const entry = document.querySelector("#entry").value;
        const response = await fetch("/", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ entry })
        });
        const data = await response.json();
        result.innerText = "Detected Mood: " + data.mood;
        document.querySelector("#entry").value = "";
    });
}

// Chart.js for dashboard
const chartElem = document.getElementById("moodChart");
if (chartElem) {
    const labels = JSON.parse(chartElem.dataset.labels);
    const moods = JSON.parse(chartElem.dataset.moods);
    new Chart(chartElem, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Mood over time",
                data: moods,
                borderColor: "#FF7F50",
                backgroundColor: "rgba(255,127,80,0.2)",
                tension: 0.3,
                fill: true,
                pointRadius: 5
            }]
        },
        options: {
            scales: {
                y: {
                    min: -1,
                    max: 1,
                    ticks: {
                        callback: function(val) {
                            return val === 1 ? "Positive" : val === 0 ? "Neutral" : "Negative";
                        }
                    }
                }
            }
        }
    });
}
