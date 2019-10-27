from __future__ import print_function
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

sessions = AudioUtilities.GetAllSessions()

for session in sessions:
	volume = session._ctl.QueryInterface(ISimpleAudioVolume)
	volume.SetMute(0, None)
	
for session in sessions:
	volume = session._ctl.QueryInterface(ISimpleAudioVolume)
	volume.SetMasterVolume(1.0, None)