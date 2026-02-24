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
        // In a real app we'd authenticate. For now, set simulated ID in local storage.
        localStorage.setItem('loggedInPatientId', id || 'P-889012');
        window.location.href = 'patient-dashboard.html';
    } else if (role === 'doctor') {
        const docId = document.getElementById('doctor-id').value;
        localStorage.setItem('loggedInDoctorId', docId || 'DR-5541');
        window.location.href = 'doctor-dashboard.html';
    }
}
