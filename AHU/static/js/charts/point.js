
const ctx3 = document.getElementById('myChart3').getContext('2d');

new Chart(ctx3, {
    type: 'bar',  // You mentioned that 'bar' type is showing, so let's stick with it
    data: {
        labels: {{ data.labels|safe }},  // Load labels from Django view data
        datasets: [{
            label: 'AHU Performance',
            data: {{ data.values|safe }},  // Load values from Django view data
            backgroundColor: 'rgba(251, 189, 65, 0.936)',
            // backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderColor: 'rgba(255, 162, 0,1)',
            // borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 2,
        }]
    },
    options: {
        scales: {
            x: {
                beginAtZero: true,
                ticks: {
                    callback: function(value, index, values) {
                        return {{ data.labels|safe }}[index];
                    }
                }
            },
            y: {
                beginAtZero: true,
                ticks: {
                    callback: function(value, index, values) {
                        return value ;
                    }
                }
            }
        }
    }
});