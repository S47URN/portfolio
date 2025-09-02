const toggleBtn = document.getElementById('themeToggle');

function setTheme(dark) {
    if (dark) {
        document.documentElement.classList.add('dark-theme');
        toggleBtn.innerHTML = '&#9789;'; // moon
    } else {
        document.documentElement.classList.remove('dark-theme');
        toggleBtn.innerHTML = '&#9728;'; // sun
    }
    localStorage.setItem('dark', dark);
}

const savedPref = localStorage.getItem('dark') === 'true';
setTheme(savedPref);

toggleBtn.addEventListener('click', () => {
    const isDark = document.documentElement.classList.contains('dark-theme');
    setTheme(!isDark);
});

// Scroll Animation
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, {
    threshold: 0.1
});

const sectionsToReveal = document.querySelectorAll('.reveal-on-scroll');
sectionsToReveal.forEach(section => {
    observer.observe(section);
});
