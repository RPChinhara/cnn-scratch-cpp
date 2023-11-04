#include <dsound.h>

// TODO: What is the best name fot file and class?
class AudioPlayer {
public:
    AudioPlayer(HWND hwnd);
    ~AudioPlayer();

    DWORD GetAudioFileSize(const char* audioFilePath);
    bool Initialize();
    bool LoadAudioData(const char* audioFilePath);
    bool PlaySound();

private:
    LPDIRECTSOUND8 pDSound;
    LPDIRECTSOUNDBUFFER pDSoundBuffer;
    WAVEFORMATEX wfx;
};