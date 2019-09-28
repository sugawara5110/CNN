//*****************************************************************************************//
//**                                                                                     **//
//**                   �@�@�@      PPMLoader�N���X                                       **//
//**                                                                                     **//
//*****************************************************************************************//

#include "PPMLoader.h"
#include <stdio.h>

PPMLoader::PPMLoader(wchar_t* pass, UINT outW, UINT outH, PPM_MODE mode) {

	sf = new SearchFile(1);
	char** str = new char* [1];
	str[0] = "ppm";
	sf->Search(pass, 0, str, 1);
	delete[]str;
	str = nullptr;
	fileNum = sf->GetFileNum(0);
	UINT modeScale = 1;
	if (mode == NORMAL)modeScale = 3;
	image = new BYTE[fileNum * outW * modeScale * outH];

	BYTE* tmpimage;
	size_t size;
	long offset;
	UINT fileCount = 0;

	for (UINT k = 0; k < fileNum; k++) {
		FILE* fp = nullptr;
		char line[200] = { 0 };//1�s�ǂݍ��ݗp
		int inW = 0;
		int inH = 0;
		int pix = 0;
		char* name = sf->GetFileName(0, k);
		fopen_s(&fp, name, "rb");
		fgets(line, sizeof(line), fp);//1�s��΂�, P6
		fgets(line, sizeof(line), fp);
		while (line[0] == '#') {
			fgets(line, sizeof(line), fp);//�R�����g���X�L�b�v
		}
		sscanf_s(line, "%d %d", &inW, &inH);
		UINT inNum = inW * 3 * inH;
		tmpimage = new BYTE[inNum];
		size = sizeof(BYTE) * inNum;
		offset = inNum;
		fgets(line, sizeof(line), fp);
		sscanf_s(line, "%d", &pix);
		//��������pixdata, RGB�̏���1byte����1�s�N�Z��3byte
		fread(tmpimage, size, 1, fp);
		//�T�C�Y�ϊ�, �O���[�X�P�[���ϊ�
		for (UINT y = 0; y < outH; y++) {
			for (UINT x = 0; x < outW; x++) {
				float scaleY = (float)inH / (float)outH;
				float scaleX = (float)inW / (float)outW;
				UINT inHeiInd = (UINT)(scaleY * y) * inW * 3;
				UINT inWidInd = (UINT)(scaleX * x) * 3;
				UINT inInd = inHeiInd + inWidInd;
				if (mode == GRAYSCALE) {
					BYTE gray = (tmpimage[inInd] + tmpimage[inInd + 1] + tmpimage[inInd + 2]) / 3;//grayscale
					image[outH * outW * fileCount + outW * y + x] = gray;
				}
				else {
					UINT imIndexst = outH * outW * modeScale * fileCount + outW * modeScale * y + x * modeScale;
					image[imIndexst] = tmpimage[inInd];
					image[imIndexst + 1] = tmpimage[inInd + 1];
					image[imIndexst + 2] = tmpimage[inInd + 2];
				}
			}
		}
		fileCount++;
		fclose(fp);
		if (tmpimage) {
			delete[] tmpimage;
			tmpimage = nullptr;
		}
	}
}

PPMLoader::~PPMLoader() {
	if (sf) {
		delete sf;
		sf = nullptr;
	}
	if (image) {
		delete[] image;
		image = nullptr;
	}
}

UINT PPMLoader::GetFileNum() {
	return fileNum;
}

BYTE *PPMLoader::GetImageArr() {
	return image;
}