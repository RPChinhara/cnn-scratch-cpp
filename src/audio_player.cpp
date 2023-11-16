#include "audio_player.h"

#pragma comment(lib, "dsound.lib")

AudioPlayer::AudioPlayer(HWND hwnd) : pDSound(nullptr), pDSoundBuffer(nullptr) {
    if (FAILED(DirectSoundCreate8(nullptr, &pDSound, nullptr))) {
        // Handle initialization error
        MessageBox(nullptr, "Failed to create DirectSound.", "Error", MB_ICONERROR);
    }

    if (FAILED(pDSound->SetCooperativeLevel(hwnd, DSSCL_PRIORITY))) {
        // Handle initialization error
        MessageBox(nullptr, "Failed to set cooperative level.", "Error", MB_ICONERROR);
    }
}

AudioPlayer::~AudioPlayer() {
    if (pDSoundBuffer) {
        pDSoundBuffer->Release();
    }

    if (pDSound) {
        pDSound->Release();
    }
}

DWORD AudioPlayer::GetAudioFileSize(const char* audioFilePath) {
    HANDLE hFile = CreateFile(audioFilePath, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        // Handle file open error
        MessageBox(nullptr, "Failed to open file.", "Error", MB_ICONERROR);
        return 0; // Return an appropriate error value
    }

    DWORD fileSize = GetFileSize(hFile, nullptr);
    CloseHandle(hFile);
    return fileSize;
}

bool AudioPlayer::Initialize() {
    // Set up the WAVEFORMATEX structure for your audio file
    wfx.wFormatTag = WAVE_FORMAT_PCM;
    wfx.nChannels = 2;
    wfx.nSamplesPerSec = 44100;
    wfx.nAvgBytesPerSec = 176400;
    wfx.nBlockAlign = 4;
    wfx.wBitsPerSample = 16;
    wfx.cbSize = 0;

    return true; // Return success or failure
}

bool AudioPlayer::LoadAudioData(const char* audioFilePath) {
    DWORD audioDataSize = GetAudioFileSize(audioFilePath);

    // Load audio data into the sound buffer
    DSBUFFERDESC dsbd = {};
    dsbd.dwSize = sizeof(dsbd);
    dsbd.dwFlags = DSBCAPS_CTRLVOLUME;
    dsbd.dwBufferBytes = audioDataSize;
    dsbd.lpwfxFormat = &wfx;

    if (FAILED(pDSound->CreateSoundBuffer(&dsbd, &pDSoundBuffer, nullptr))) {
        // Handle sound buffer creation error
        MessageBox(nullptr, "Failed to create sound buffer.", "Error", MB_ICONERROR);
        return false;
    }

    // Fill pDSoundBuffer with audio data from your audio file
    void* pAudioData = nullptr;
    DWORD audioDataSizeLocked = 0;

    if (FAILED(pDSoundBuffer->Lock(0, audioDataSize, &pAudioData, &audioDataSizeLocked, nullptr, nullptr, 0))) {
        // Handle locking error
        MessageBox(nullptr, "Failed to lock sound buffer.", "Error", MB_ICONERROR);
        return false;
    }

    // Copy audio data from your file into pAudioData
    // Make sure to set audioDataSizeLocked to the size of the data copied

    pDSoundBuffer->Unlock(pAudioData, audioDataSizeLocked, nullptr, 0);

    return true; // Return success or failure
}

bool AudioPlayer::PlaySound() {
    if (pDSoundBuffer) {
        if (FAILED(pDSoundBuffer->SetCurrentPosition(0))) {
            // Handle position setting error
            MessageBox(nullptr, "Failed to set sound buffer position.", "Error", MB_ICONERROR);
            return false;
        }

        if (FAILED(pDSoundBuffer->Play(0, 0, 0))) {
            // Handle playback error
            MessageBox(nullptr, "Failed to play sound buffer.", "Error", MB_ICONERROR);
            return false;
        }
        return true; // Sound playback succeeded
    }

    return false; // Sound buffer not created
}