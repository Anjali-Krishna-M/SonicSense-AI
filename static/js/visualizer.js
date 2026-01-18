// Simple console log to prove JS is connected
console.log("VibeCheck Audio Engine: Ready");

document.addEventListener('DOMContentLoaded', () => {
    const audioPlayer = document.querySelector('audio');
    if(audioPlayer) {
        audioPlayer.volume = 0.5; // Start at 50% volume
    }
});