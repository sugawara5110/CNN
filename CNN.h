//*****************************************************************************************//
//**                                                                                     **//
//**                                    CNN                                              **//
//**                                                                                     **//
//*****************************************************************************************//
#ifndef Class_CNN_Header
#define Class_CNN_Header

#include "../Common/Direct3DWrapper/Dx_NN.h"

class CNN;
class Convolution;
class Pooling;

class GradCAM :public DxGradCAM {

protected:
	PolygonData2D dgc;

public:
	GradCAM(UINT srcWid, UINT srcHei, UINT SizeFeatureMapW, UINT SizeFeatureMapH, UINT NumGradientEl,
		UINT NumFil, UINT inputsetnum);
	void Draw(float x, float y);
};

class Affine :public DxNeuralNetwork {

protected:
	friend CNN;
	Convolution* errCn = nullptr;
	Pooling* errPo = nullptr;
	Convolution* inCn = nullptr;
	Pooling* inPo = nullptr;
	PolygonData2D dnn;
	UINT NumFilter;

public:
	Affine(ActivationName activationName, ActivationName topActivationName, UINT inW, UINT inH, UINT* numNode,
		int depth, UINT split, UINT inputsetnum);
	void Draw(float x, float y);
	void InConnection();
	void ErrConnection();
	void TestConnection();
	void DetectionConnection(UINT SearchNum);
};

class Pooling :public DxPooling {

protected:
	friend CNN;
	Affine* inAf = nullptr;
	Convolution* inCn = nullptr;
	Affine* errAf = nullptr;
	Convolution* errCn = nullptr;
	PolygonData2D dpo;
	UINT NumFilter;

public:
	Pooling(UINT width, UINT height, UINT poolNum, UINT inputsetnum);
	void Draw(float x, float y);
	void InConnection();
	void ErrConnection();
	void TestConnection();
	void DetectionConnection(UINT SearchNum, bool GradCAM_ON = false);
};

class Convolution :public DxConvolution {

protected:
	friend CNN;
	Affine* inAf = nullptr;
	Pooling* inPo = nullptr;
	Convolution* inCn = nullptr;
	Affine* errAf = nullptr;
	Convolution* errCn = nullptr;
	Pooling* errPo = nullptr;
	PolygonData2D dcn;
	UINT NumFilter;

public:
	Convolution(ActivationName activationName, UINT width, UINT height, UINT filNum, UINT inputsetnum, UINT elnumwid, UINT filstep);
	void Draw(float x, float y);
	void InConnection();
	void ErrConnection();
	void TestConnection();
	void DetectionConnection(UINT SearchNum, bool GradCAM_ON = false);
};

enum LayerName {
	CONV,
	POOL,
	AFFINE
};

struct Layer {
	UINT maxThread;//最大スレッド数(全レイヤー共通)
	UINT mapWid;//検出範囲wid
	UINT mapHei;//検出範囲hei
	LayerName layerName;
	UINT NumFilter;
	UINT NumConvFilterWid;//畳み込みフィルターサイズ
	UINT NumConvFilterSlide;//畳み込みフィルタースライド量
	UINT numNode[MAX_DEPTH_NUM - 1];//Affineのノード数(入力層除き)
	UINT NumDepthNotInput;//Affineの深さ(入力除く)
	ActivationName acName;//活性化関数名
	ActivationName topAcName;//活性化関数名最終出力(Affineのみ)
};

class CNN {

protected:
	UINT NumConv = 0;
	UINT NumPool = 0;
	Convolution** cn = nullptr;
	Pooling** po = nullptr;
	Affine* nn = nullptr;
	GradCAM* gc = nullptr;
	UINT layerSize = 0;
	LayerName firstLayer;
	LayerName endLayer;

	CNN() {}

public:
	CNN(UINT srcW, UINT srcH, Layer* layer, UINT layerSize);
	~CNN();
	void Detection(UINT SearchNum);
	void DetectionGradCAM(UINT SearchNum, UINT srcMapWid, UINT mapslide);
	void SetLearningLate(float nn, float cn);
	void Training();
	void TrainingFp();
	void TrainingBp();
	float GetcrossEntropyError();
	float GetcrossEntropyErrorTest();
	void Test();
	void TrainingDraw();
	void GradCAMDraw();
	void GetOutput(float* out, UINT inputsetInd = 0);
	float GetOutputEl(UINT ElNum, UINT inputsetInd = 0);
	ID3D12Resource* GetOutputResource();
	void SetTargetEl(float el, UINT ElNum);
	void FirstInput(float el, UINT ElNum, UINT inputsetInd = 0);
	void SetPixel3ch(ID3D12Resource* pi);
	void SetPixel3ch(BYTE* pi);
	void SaveData();
	void LoadData();
};

#endif