//*****************************************************************************************//
//**                                                                                     **//
//**                   　　　      PPMLoaderクラス                                       **//
//**                                                                                     **//
//*****************************************************************************************//

#ifndef Class_PPMLoader_Header
#define Class_PPMLoader_Header

#include "../Common/SearchFile\SearchFile.h"

enum PPM_MODE {
	NORMAL,
	GRAYSCALE
};

class PPMLoader {

private:
	SearchFile* sf = nullptr;
	UINT fileNum = 0;
	BYTE* image = nullptr;

	PPMLoader() {}

public:
	PPMLoader(wchar_t* pass, UINT outW, UINT outH, PPM_MODE mode);
	~PPMLoader();
	UINT GetFileNum();
	BYTE* GetImageArr();
};

#endif
