//*****************************************************************************************//
//**                                                                                     **//
//**                                    CNN                                              **//
//**                                                                                     **//
//*****************************************************************************************//

#include "CNN.h"

GradCAM::GradCAM(UINT srcWid, UINT srcHei, UINT SizeFeatureMapW, UINT SizeFeatureMapH, UINT NumGradientEl, UINT NumFil, UINT inputsetnum) :
	DxGradCAM(SizeFeatureMapW, SizeFeatureMapH, NumGradientEl, NumFil, inputsetnum) {

	ComCreate(srcWid, srcHei, 1.0f);
	dgc.SetCommandList(0);
	dgc.GetVBarray2D(1);
	dgc.TextureInit(srcWid, srcHei);
	dgc.TexOn();
	dgc.CreateBox(0.0f, 0.0f, 0.0f, 0.1f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, TRUE, TRUE);
}

void GradCAM::Draw(float x, float y) {
	dgc.Update(x, y, 0.8f, 1.0f, 1.0f, 1.0f, 1.0f, 300.0f, 200.0f);
	dgc.CopyResource(GetPixel(), GetNNTextureResourceStates());
	dgc.Draw();
}

Affine::Affine(ActivationName activationName, OptimizerName optName, ActivationName topActivationName, UINT inW, UINT inH, UINT* numNode,
	int depth, UINT split, UINT inputsetnum) :
	DxNeuralNetwork(numNode, depth, split, inputsetnum) {

	NumFilter = split;
	ComCreate(activationName, optName, topActivationName);
	SetActivationAlpha(0.05f);
	CreareNNTexture(inW, inH, NumFilter);
	dnn.SetCommandList(0);
	dnn.GetVBarray2D(1);
	dnn.TextureInit(inW, inH * NumFilter);
	dnn.TexOn();
	dnn.CreateBox(0.0f, 0.0f, 0.0f, 0.1f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, TRUE, TRUE);
}

void Affine::Draw(float x, float y) {
	dnn.Update(x, y, 0.8f, 1.0f, 1.0f, 1.0f, 1.0f, 20.0f, 20.0f * NumFilter);
	dnn.CopyResource(GetNNTextureResource(), GetNNTextureResourceStates());
	dnn.Draw();
}

void Affine::InConnection() {
	TrainingFp();
	if (inCn) {
		inCn->SetInputResource(GetOutputResource());
		inCn->InConnection();
	}
	if (inPo) {
		inPo->SetInputResource(GetOutputResource());
		inPo->InConnection();
	}
}

void Affine::ErrConnection(bool update) {
	if (update)
		TrainingBp();
	else
		TrainingBpNoWeightUpdate();
	if (errPo) {
		errPo->SetInErrorResource(GetOutErrorResource());
		errPo->ErrConnection(update);
	}
	if (errCn) {
		errCn->SetInErrorResource(GetOutErrorResource());
		errCn->ErrConnection(update);
	}
}

void Affine::TestConnection() {
	Test();
	if (inCn) {
		inCn->SetInputResource(GetOutputResource());
		inCn->TestConnection();
	}
	if (inPo) {
		inPo->SetInputResource(GetOutputResource());
		inPo->TestConnection();
	}
}

void Affine::DetectionConnection(UINT SearchNum) {
	Query(SearchNum);
	if (inCn) {
		inCn->SetInputResource(GetOutputResource());
		inCn->DetectionConnection(SearchNum);
	}
	if (inPo) {
		inPo->SetInputResource(GetOutputResource());
		inPo->DetectionConnection(SearchNum);
	}
}

Pooling::Pooling(UINT width, UINT height, UINT poolNum, UINT inputsetnum) :
	DxPooling(width, height, poolNum, inputsetnum) {

	NumFilter = poolNum;
	ComCreate();
	UINT wid = GetOutWidth();
	UINT hei = GetOutHeight();
	CreareNNTexture(wid, hei, NumFilter);
	dpo.SetCommandList(0);
	dpo.GetVBarray2D(1);
	dpo.TextureInit(wid, hei * NumFilter);
	dpo.TexOn();
	dpo.CreateBox(0.0f, 0.0f, 0.0f, 0.1f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, TRUE, TRUE);
}

void Pooling::Draw(float x, float y) {
	dpo.Update(x, y, 0.8f, 1.0f, 1.0f, 1.0f, 1.0f, 20.0f, 20.0f * NumFilter);
	dpo.CopyResource(GetNNTextureResource(), GetNNTextureResourceStates());
	dpo.Draw();
}

void Pooling::InConnection() {
	Query();
	if (inCn) {
		inCn->SetInputResource(GetOutputResource());
		inCn->InConnection();
	}
	if (inAf) {
		inAf->SetInputResource(GetOutputResource());
		inAf->InConnection();
	}
}

void Pooling::ErrConnection(bool update) {
	Training();
	if (errCn) {
		errCn->SetInErrorResource(GetOutErrorResource());
		errCn->ErrConnection(update);
	}
	if (errAf) {
		errAf->SetInErrorResource(GetOutErrorResource());
		errAf->ErrConnection(update);
	}
}

void Pooling::TestConnection() {
	Test();
	if (inCn) {
		inCn->SetInputResource(GetOutputResource());
		inCn->TestConnection();
	}
	if (inAf) {
		inAf->SetInputResource(GetOutputResource());
		inAf->TestConnection();
	}
}

void Pooling::DetectionConnection(UINT SearchNum, bool GradCAM_ON) {
	Detection(SearchNum);
	if (inCn) {
		inCn->SetInputResource(GetOutputResource());
		inCn->DetectionConnection(SearchNum);
	}
	if (inAf) {
		inAf->SetInputResource(GetOutputResource());
		if (GradCAM_ON)inAf->DetectionConnection(SearchNum);
	}
}

Convolution::Convolution(ActivationName activationName, UINT width, UINT height, UINT filNum, bool DeConvolutionMode, UINT inputsetnum, UINT elnumwid, UINT filstep) :
	DxConvolution(width, height, filNum, DeConvolutionMode, inputsetnum, elnumwid, filstep) {

	NumFilter = filNum;
	ComCreate(activationName);
	SetActivationAlpha(0.05f);
	CreareNNTexture(elnumwid, elnumwid, NumFilter);
	dcn.SetCommandList(0);
	dcn.GetVBarray2D(1);
	dcn.TextureInit(elnumwid, elnumwid * NumFilter);
	dcn.TexOn();
	dcn.CreateBox(0.0f, 0.0f, 0.0f, 0.1f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, TRUE, TRUE);
}

void Convolution::Draw(float x, float y) {
	dcn.Update(x, y, 0.8f, 1.0f, 1.0f, 1.0f, 1.0f, 20.0f, 20.0f * NumFilter);
	dcn.CopyResource(GetNNTextureResource(), GetNNTextureResourceStates());
	dcn.Draw();
}

void Convolution::InConnection() {
	Query();
	if (inCn) {
		inCn->SetInputResource(GetOutputResource());
		inCn->InConnection();
	}
	if (inPo) {
		inPo->SetInputResource(GetOutputResource());
		inPo->InConnection();
	}
	if (inAf) {
		inAf->SetInputResource(GetOutputResource());
		inAf->InConnection();
	}
}

void Convolution::ErrConnection(bool update) {
	if (update)
		Training();
	else
		BackPropagationNoWeightUpdate();
	if (errCn) {
		errCn->SetInErrorResource(GetOutErrorResource());
		errCn->ErrConnection(update);
	}
	if (errPo) {
		errPo->SetInErrorResource(GetOutErrorResource());
		errPo->ErrConnection(update);
	}
	if (errAf) {
		errAf->SetInErrorResource(GetOutErrorResource());
		errAf->ErrConnection(update);
	}
}

void Convolution::TestConnection() {
	Test();
	if (inCn) {
		inCn->SetInputResource(GetOutputResource());
		inCn->TestConnection();
	}
	if (inPo) {
		inPo->SetInputResource(GetOutputResource());
		inPo->TestConnection();
	}
	if (inAf) {
		inAf->SetInputResource(GetOutputResource());
		inAf->TestConnection();
	}
}

void Convolution::DetectionConnection(UINT SearchNum, bool GradCAM_ON) {
	Detection(SearchNum);
	if (inCn) {
		inCn->SetInputResource(GetOutputResource());
		inCn->DetectionConnection(SearchNum);
	}
	if (inPo) {
		inPo->SetInputResource(GetOutputResource());
		inPo->DetectionConnection(SearchNum);
	}
	if (inAf) {
		inAf->SetInputResource(GetOutputResource());
		if (!GradCAM_ON)inAf->DetectionConnection(SearchNum);
	}
}

CNN::CNN(UINT srcW, UINT srcH, Layer* layer, UINT layersize) {
	NumConv = 0;
	NumPool = 0;
	layerSize = layersize;
	UINT affIndex = 0;
	//レイヤーカウント, AffineIndex探索
	for (UINT i = 0; i < layerSize; i++) {
		switch (layer[i].layerName) {
		case CONV:
			NumConv++;
			break;
		case POOL:
			NumPool++;
			break;
		case AFFINE:
			affIndex = i;
			break;
		}
	}

	firstLayer = layer[0].layerName;
	endLayer = layer[layersize - 1].layerName;

	cn = new Convolution * [NumConv];
	po = new Pooling * [NumPool];
	UINT NumDepth = 0;

	if (layer[affIndex].NumDepthNotInput > MAX_DEPTH_NUM - 1)NumDepth = MAX_DEPTH_NUM;
	else
		NumDepth = layer[affIndex].NumDepthNotInput + 1;

	//レイヤー生成
	UINT convCnt = 0;
	UINT poolCnt = 0;
	UINT wid = layer[0].mapWid;
	UINT hei = layer[0].mapHei;
	UINT numN[MAX_DEPTH_NUM];
	for (UINT i = 0; i < layerSize; i++) {
		switch (layer[i].layerName) {
		case CONV:
			cn[convCnt] = new Convolution(layer[i].acName, wid, hei, layer[i].NumFilter, layer[i].DeConvolutionMode, layer[0].maxThread,
				layer[i].NumConvFilterWid, layer[i].NumConvFilterSlide);
			wid = cn[convCnt]->GetOutWidth();
			hei = cn[convCnt++]->GetOutHeight();
			if (convCnt == NumConv) {
				gc = new GradCAM(srcW, srcH, wid, hei, layer[i].NumConvFilterWid * layer[i].NumConvFilterWid,
					layer[i].NumFilter, layer[0].maxThread);
			}
			break;
		case POOL:
			po[poolCnt] = new Pooling(wid, hei, layer[i].NumFilter, layer[0].maxThread);
			wid = po[poolCnt]->GetOutWidth();
			hei = po[poolCnt++]->GetOutHeight();
			break;
		case AFFINE:
			numN[0] = wid * hei;
			for (UINT i1 = 1; i1 < NumDepth; i1++)numN[i1] = layer[i].numNode[i1 - 1];
			nn = new Affine(layer[i].acName, layer[i].optName, layer[i].topAcName, wid, hei, numN, NumDepth, layer[i].NumFilter,
				layer[0].maxThread);
			wid = layer[i].topNodeWid;
			hei = numN[NumDepth - 1] / wid;
			break;
		}
	}

	//レイヤー接続
	convCnt = 0;
	poolCnt = 0;
	for (UINT i = 0; i < layerSize; i++) {
		switch (layer[i].layerName) {
		case CONV:
			//誤差側接続
			if (i > 0) {
				if (layer[i - 1].layerName == CONV)cn[convCnt]->errCn = cn[convCnt - 1];
				if (layer[i - 1].layerName == POOL)cn[convCnt]->errPo = po[poolCnt - 1];
				if (layer[i - 1].layerName == AFFINE)cn[convCnt]->errAf = nn;
			}
			//入力側接続
			if (i < layerSize - 1) {
				if (layer[i + 1].layerName == CONV)cn[convCnt]->inCn = cn[convCnt + 1];
				if (layer[i + 1].layerName == POOL)cn[convCnt]->inPo = po[poolCnt];
				if (layer[i + 1].layerName == AFFINE)cn[convCnt]->inAf = nn;
			}
			convCnt++;
			break;
		case POOL:
			//誤差側接続
			if (layer[i - 1].layerName == CONV)po[poolCnt]->errCn = cn[convCnt - 1];
			if (layer[i - 1].layerName == AFFINE)po[poolCnt]->errAf = nn;
			//入力側接続
			if (layer[i + 1].layerName == CONV)po[poolCnt]->inCn = cn[convCnt];
			if (layer[i + 1].layerName == AFFINE)po[poolCnt]->inAf = nn;
			poolCnt++;
			break;
		case AFFINE:
			//誤差側接続
			if (i > 0) {
				if (layer[i - 1].layerName == CONV)nn->errCn = cn[convCnt - 1];
				if (layer[i - 1].layerName == POOL)nn->errPo = po[poolCnt - 1];
			}
			//入力側接続
			if (i < layerSize - 1) {
				if (layer[i + 1].layerName == CONV)nn->inCn = cn[convCnt];
				if (layer[i + 1].layerName == POOL)nn->inPo = po[poolCnt];
			}
			break;
		}
	}
}

CNN::~CNN() {
	for (UINT i = 0; i < NumConv; i++)S_DELETE(cn[i]);
	ARR_DELETE(cn);
	for (UINT i = 0; i < NumPool; i++)S_DELETE(po[i]);
	ARR_DELETE(po);
	S_DELETE(nn);
	S_DELETE(gc);
}

void CNN::Detection(UINT SearchNum) {
	switch (firstLayer) {
	case CONV:
		cn[0]->DetectionConnection(SearchNum);
		break;
	case AFFINE:
		nn->DetectionConnection(SearchNum);
		break;
	}
}

void CNN::DetectionGradCAM(UINT SearchNum, UINT srcMapWid, UINT mapslide) {
	cn[0]->DetectionConnection(SearchNum, true);
	gc->SetFeatureMap(cn[NumConv - 1]->GetOutputResource());//最終Convの出力を記録()
	nn->SetTargetEl(0.99f, 0);
	nn->QueryAndBackPropagation(SearchNum);//フィルター更新無しの逆伝播
	if (nn->errPo) {
		nn->errPo->SetInErrorResource(nn->GetOutErrorResource());
		nn->errPo->Training();
		nn->errPo->errCn->SetInErrorResource(nn->errPo->GetOutErrorResource());
		nn->errPo->errCn->BackPropagationNoWeightUpdate();//フィルター更新無しの逆伝播
	}
	if (nn->errCn) {
		nn->errCn->SetInErrorResource(nn->GetOutErrorResource());
		nn->errCn->BackPropagationNoWeightUpdate();
	}

	gc->SetGradient(cn[NumConv - 1]->GetGradient());
	gc->ComGAP();
	gc->ComGradCAM(SearchNum);
	gc->GradCAMSynthesis(srcMapWid, srcMapWid, mapslide);
}

void CNN::SetLearningLate(float nN, float cN, float cnB) {
	for (UINT i = 0; i < NumConv; i++) {
		cn[i]->setOptimizerParameterFil(cN);
		cn[i]->setOptimizerParameterBias(cnB);
	}
	nn->setOptimizerParameter(nN);
}

void CNN::setOptimizerParameter(float LearningRateNN, float AttenuationRate1NN,
	float AttenuationRate2NN, float DivergencePreventionNN,
	float LearningRateCN, float AttenuationRate1CN,
	float AttenuationRate2CN, float DivergencePreventionCN,
	float LearningRateCNB, float AttenuationRate1CNB,
	float AttenuationRate2CNB, float DivergencePreventionCNB) {

	for (UINT i = 0; i < NumConv; i++) {
		cn[i]->setOptimizerParameterFil(LearningRateCN, AttenuationRate1CN,
			AttenuationRate2CN, DivergencePreventionCN);
		cn[i]->setOptimizerParameterBias(LearningRateCNB, AttenuationRate1CNB,
			AttenuationRate2CNB, DivergencePreventionCNB);
	}
	nn->setOptimizerParameter(LearningRateNN, AttenuationRate1NN,
		AttenuationRate2NN, DivergencePreventionNN);
}

void CNN::SetActivationAlpha(float nN, float cN) {
	for (UINT i = 0; i < NumConv; i++)cn[i]->SetActivationAlpha(cN);
	nn->SetActivationAlpha(nN);
}

void CNN::SetdropThreshold(float* dropNN, float dropCN) {
	for (UINT i = 0; i < NumConv; i++)cn[i]->SetdropThreshold(dropCN);
	nn->SetdropThreshold(dropNN);
}

void CNN::Training() {
	TrainingFp();
	TrainingBp();
}

void CNN::TrainingFp() {
	switch (firstLayer) {
	case CONV:
		cn[0]->InConnection();
		break;
	case AFFINE:
		nn->InConnection();
		break;
	}
}

void CNN::TrainingBp() {
	switch (endLayer) {
	case CONV:
		cn[NumConv - 1]->ErrConnection(true);
		break;
	case POOL:
		po[NumPool - 1]->ErrConnection(true);
		break;
	case AFFINE:
		nn->ErrConnection(true);
		break;
	}
}

void CNN::TrainingBpNoUpdate() {
	switch (endLayer) {
	case CONV:
		cn[NumConv - 1]->ErrConnection(false);
		break;
	case POOL:
		po[NumPool - 1]->ErrConnection(false);
		break;
	case AFFINE:
		nn->ErrConnection(false);
		break;
	}
}

float CNN::GetcrossEntropyError() {
	return nn->GetcrossEntropyError();
}

float CNN::GetcrossEntropyErrorTest() {
	return nn->GetcrossEntropyErrorTest();
}

void CNN::Test() {
	switch (firstLayer) {
	case CONV:
		cn[0]->TestConnection();
		break;
	case AFFINE:
		nn->TestConnection();
		break;
	}
}

void CNN::TrainingDraw(float x0, float y) {
	float x = x0;
	for (UINT i = 0; i < NumConv; i++) {
		cn[i]->Draw(x, y);
		x += 20.0f;
	}
	for (UINT i = 0; i < NumPool; i++) {
		po[i]->Draw(x, y);
		x += 20.0f;
	}
	nn->Draw(x, y);
}

void CNN::GradCAMDraw(float x, float y) {
	gc->Draw(x, y);
}

float CNN::GetOutputEl(UINT ElNum, UINT inputsetInd) {
	UINT numFil = 0;
	UINT oneFilEl = 0;
	UINT numEl = 0;
	switch (endLayer) {
	case CONV:
		oneFilEl = cn[NumConv - 1]->GetOutWidth() * cn[NumConv - 1]->GetOutHeight();
		numFil = ElNum / oneFilEl;
		numEl = ElNum % oneFilEl;
		return cn[NumConv - 1]->OutputEl(numFil, numEl, inputsetInd);
		break;
	case POOL:
		oneFilEl = po[NumPool - 1]->GetOutWidth() * po[NumPool - 1]->GetOutHeight();
		numFil = ElNum / oneFilEl;
		numEl = ElNum % oneFilEl;
		return po[NumPool - 1]->OutputEl(numFil, numEl, inputsetInd);
		break;
	case AFFINE:
		return nn->GetOutputEl(ElNum, inputsetInd);
		break;
	}
	return 0.0f;
}

ID3D12Resource* CNN::GetOutputResource() {
	switch (endLayer) {
	case CONV:
		return cn[NumConv - 1]->GetOutputResource();
		break;
	case POOL:
		return po[NumPool - 1]->GetOutputResource();
		break;
	case AFFINE:
		return nn->GetOutputResource();
		break;
	}
	return nullptr;
}

ID3D12Resource* CNN::GetOutErrResource() {
	switch (firstLayer) {
	case CONV:
		return cn[0]->GetOutErrorResource();
		break;
	case POOL:
		return po[0]->GetOutErrorResource();
		break;
	case AFFINE:
		return nn->GetOutErrorResource();
		break;
	}
	return nullptr;
}

void CNN::SetInErrResource(ID3D12Resource* res) {
	switch (endLayer) {
	case CONV:
		cn[NumConv - 1]->SetInErrorResource(res);
		break;
	case POOL:
		po[NumPool - 1]->SetInErrorResource(res);
		break;
	case AFFINE:
		nn->SetInErrorResource(res);
		break;
	}
}

void CNN::SetTargetEl(float el, UINT ElNum) {
	nn->SetTargetEl(el, ElNum);
}

void CNN::FirstInput(float el, UINT ElNum, UINT inputsetInd) {
	switch (firstLayer) {
	case CONV:
		cn[0]->FirstInput(el, ElNum, inputsetInd);
		break;
	case AFFINE:
		nn->FirstInput(el, ElNum, inputsetInd);
		break;
	}
}

void CNN::InputArray(float* inArr, UINT arrNum, UINT inputsetInd) {
	switch (firstLayer) {
	case CONV:
		cn[0]->Input(inArr, arrNum, inputsetInd);
		break;
	case AFFINE:
		nn->InputArray(inArr, arrNum, inputsetInd);
		break;
	}
}

void CNN::InputArrayEl(float el, UINT arrNum, UINT ElNum, UINT inputsetInd) {
	switch (firstLayer) {
	case CONV:
		cn[0]->InputEl(el, arrNum, ElNum, inputsetInd);
		break;
	case AFFINE:
		nn->InputArrayEl(el, arrNum, ElNum, inputsetInd);
		break;
	}
}

void CNN::SetPixel3ch(ID3D12Resource* pi) {
	gc->SetPixel3ch(pi);
}

void CNN::SetPixel3ch(BYTE* pi) {
	gc->SetPixel3ch(pi);
}

void CNN::SaveData() {
	for (UINT i = 0; i < NumConv; i++)cn[i]->SaveData(i);
	nn->SaveData();
}

void CNN::LoadData() {
	for (UINT i = 0; i < NumConv; i++)cn[i]->LoadData(i);
	nn->LoadData();
}