function switchTab(role) {
    const tabs = document.querySelectorAll('.tab-btn');
    const forms = document.querySelectorAll('.login-form');

    tabs.forEach(tab => tab.classList.remove('active'));
    forms.forEach(form => form.classList.remove('active-form'));

    if (role === 'patient') {
        tabs[0].classList.add('active');
        document.getElementById('patient-login-form').classList.add('active-form');
    } else {
        tabs[1].classList.add('active');
        document.getElementById('doctor-login-form').classList.add('active-form');
    }
}

function handleLogin(event, role) {
    event.preventDefault(); // Prevent form from submitting normally

    if (role === 'patient') {
        const id = document.getElementById('patient-id').value;
        const pass = document.getElementById('patient-password').value;
        if (id === 'P-889012' && pass === 'demo123') {
            localStorage.setItem('loggedInPatientId', id);
            window.location.href = 'patient-dashboard.html?v=2';
        } else {
            alert("Invalid Patient Credentials. Try P-889012 / demo123");
        }
    } else if (role === 'doctor') {
        const docId = document.getElementById('doctor-id').value;
        const pass = document.getElementById('doctor-password').value;
        if (docId === 'DR-5541' && pass === 'doctor123') {
            localStorage.setItem('loggedInDoctorId', docId);
            window.location.href = 'doctor-dashboard.html';
        } else {
            alert("Invalid Doctor Credentials. Try DR-5541 / doctor123");
        }
    }
}
