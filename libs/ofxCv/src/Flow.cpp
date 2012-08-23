#include "ofxCv/Flow.h"

namespace ofxCv {
	
	using namespace cv;
	
#pragma mark FLOW IMPLEMENTATION
	Flow::Flow(){
		hasFlow = false;
		last.setUseTexture(false);
		curr.setUseTexture(false);
	}
	
	Flow::~Flow(){
	}
	
	//call with two images
	void Flow::calcOpticalFlow(ofBaseHasPixels& lastImage, ofBaseHasPixels& currentImage){
		calcOpticalFlow(lastImage.getPixelsRef(), currentImage.getPixelsRef());
	}
	
	void Flow::calcOpticalFlow(ofPixelsRef lastImage, ofPixelsRef currentImage){
		last.setFromPixels(lastImage);
		last.setImageType(OF_IMAGE_GRAYSCALE); //force to gray
		
		calcFlow(); //will call concrete implementation
		hasFlow = true;
		
		curr.setFromPixels(currentImage);
		curr.setImageType(OF_IMAGE_GRAYSCALE);
	}
	
	//you can add subsequent images this way without having to store 
	//the previous one yourself
	void Flow::calcOpticalFlow(ofBaseHasPixels& nextImage){
		calcOpticalFlow(nextImage.getPixelsRef());
	}
	
	void Flow::calcOpticalFlow(ofPixelsRef nextImage){
		curr.setFromPixels(nextImage);
		curr.setImageType(OF_IMAGE_GRAYSCALE);
		
		if(last.isAllocated()){
			calcFlow(); //will call concrete implementation
			hasFlow = true;
		}
		
		last.setFromPixels(curr.getPixelsRef());
	}
	
	void Flow::draw(){
		if(hasFlow) {
			drawFlow(ofRectangle(0,0, last.getWidth(), last.getHeight() ), 0, 0);
		}
	}
	void Flow::draw(float x, float y){
		if(hasFlow){
			drawFlow(ofRectangle(x,y,last.getWidth(),last.getHeight()), 0, 0);
		}
	}
	void Flow::draw(float x, float y, float width, float height){
		if(hasFlow){
			drawFlow(ofRectangle(x,y,width,height), 0, 0);
		}
	}
	void Flow::draw(float x, float y, float width, float height, float x2, float y2){
		if(hasFlow){
			drawFlow(ofRectangle(x,y,width,height), x2, y2);
		}
	}
	void Flow::draw(ofRectangle rect){
		if(hasFlow){
			drawFlow(rect, 0, 0);
		}
	}
    int Flow::getWidth()  { return 0; }
    int Flow::getHeight() { return 0; }
    
	
#pragma mark PYRLK IMPLEMENTATION
	FlowPyrLK::FlowPyrLK(){
	}
	
	FlowPyrLK::~FlowPyrLK(){
	}
	
	void FlowPyrLK::setWindowSize(int winsize){
		this->windowSize = winsize;
	}
	void FlowPyrLK::setMaxLevel(int maxLevel){
		this->maxLevel = maxLevel;
	}
	void FlowPyrLK::setMaxFeatures(int maxFeatures){
		this->maxFeatures = maxFeatures;
	}
	void FlowPyrLK::setQualityLevel(float qualityLevel){
		this->qualityLevel = qualityLevel;
	}
	void FlowPyrLK::setMinDistance(int minDistance){
		this->minDistance = minDistance;
	}
	
	void FlowPyrLK::calcFlow(){
		prevPts.clear();
		goodFeaturesToTrack(
												toCv(last),
												prevPts,
												maxFeatures,
												qualityLevel,
												minDistance
												);
		
		vector<uchar> status;
		vector<float> err;
		calcOpticalFlowPyrLK(
												 toCv(last),
												 toCv(curr),
												 prevPts,
												 nextPts,
												 status,
												 err,
												 
												 cv::Size(windowSize, windowSize),
												 maxLevel
												 );
	}
	
    int FlowPyrLK::getWidth() {
        return last.getWidth();
    }
    int FlowPyrLK::getHeight() {
        return last.getHeight();
    }
    
	vector<ofPoint> FlowPyrLK::getFeatures(){
		return toOf(prevPts).getVertices();
	}
	
	void FlowPyrLK::drawFlow(ofRectangle rect, float x2, float y2) {
		ofVec2f offset(rect.x,rect.y);
		ofVec2f scale(rect.width/last.getWidth(),rect.height/last.getHeight());
		ofVec2f otherOffset(x2, y2);
		ofPushStyle();
		srand(0);
		for(int i = 0; i < prevPts.size(); i++) {
			ofColor color = ofColor::fromHex(rand() % 0xffffff);
			ofSetColor(color);
			ofLine(toOf(prevPts[i])*scale+offset, toOf(nextPts[i])*scale+offset + otherOffset);

		}
		ofPopStyle();
	}
	
	std::vector<Point2f> FlowPyrLK::getPointsPrev() {
		// WARNING: not a copy
		return prevPts;
	}

	std::vector<Point2f> FlowPyrLK::getPointsNext() {
		// WARNING: not a copy
		return nextPts;
	}


#pragma mark FARNEBACK IMPLEMENTATION
	FlowFarneback::FlowFarneback(){
	}
	
	FlowFarneback::~FlowFarneback(){
	}
	
	void FlowFarneback::setPyramidScale(float scale){
		if(scale < 0.0 || scale > 1.0){
			ofLogWarning("ofxCvFlowFarneback -- Warning setting scale to a number outside of 0 - 1");
		}
		this->pyramidScale = scale;
	}
	
	void FlowFarneback::setNumLevels(int levels){
		this->numLevels = levels;
	}
	void FlowFarneback::setWindowSize(int winsize){
		this->windowSize = winsize;
	}
	void FlowFarneback::setNumIterations(int interations){
		this->numIterations = interations;
	}
	void FlowFarneback::setPolyN(int polyN){
		this->polyN = polyN;
	}
	void FlowFarneback::setPolySigma(float polySigma){
		this->polySigma = polySigma;
	}	
	void FlowFarneback::setUseGaussian(bool gaussian){
		this->farnebackGaussian = gaussian;
	}
	
	std::vector<Point2f> FlowFarneback::getPointsPrev() {
		std::vector<cv::Point2f> ret;

		// see drawFlow for stepSize value
		int stepSize = 4;
		for (int y = 0; y < flow.rows; y += stepSize) {
			for (int x = 0; x < flow.cols; x += stepSize) {
				ret.push_back(cv::Point2f(x, y));
			}
		}

		return ret;
	}

	std::vector<Point2f> FlowFarneback::getPointsNext() {
		std::vector<cv::Point2f> ret;

		// see drawFlow for stepSize value
		int stepSize = 4;
		for (int y = 0; y < flow.rows; y += stepSize) {
			for (int x = 0; x < flow.cols; x += stepSize) {
				ofVec2f vec = getFlowPosition(x, y);
				ret.push_back(toCv(vec));
			}
		}

		return ret;
	}

	void FlowFarneback::calcFlow(){
		int flags = OPTFLOW_USE_INITIAL_FLOW;
		flags |= farnebackGaussian ? OPTFLOW_FARNEBACK_GAUSSIAN : 0;
		
		calcOpticalFlowFarneback(
														 toCv(last),
														 toCv(curr),
														 flow,
														 
														 pyramidScale,
														 numLevels,
														 windowSize,
														 numIterations,
														 polyN,
														 polySigma,
														 flags
														 );
	}
	
	ofVec2f FlowFarneback::getFlowOffset(int x, int y){
		const Vec2f& vec = flow.at<Vec2f>(y, x);
		return ofVec2f(vec[0], vec[1]);
	}
	ofVec2f FlowFarneback::getFlowPosition(int x, int y){
		const Vec2f& vec = flow.at<Vec2f>(y, x);
		return ofVec2f(x + vec[0], y + vec[1]);
	}
	ofVec2f FlowFarneback::getTotalFlow(){
		return getTotalFlowInRegion(ofRectangle(0,0,flow.cols, flow.rows));
	}
	ofVec2f FlowFarneback::getAverageFlow(){
		return getAverageFlowInRegion(ofRectangle(0,0,flow.cols,flow.rows));
	}
	
	ofVec2f FlowFarneback::getAverageFlowInRegion(ofRectangle rect){
		return getTotalFlowInRegion(rect)/(rect.width*rect.height);
	}
	
	ofVec2f FlowFarneback::getTotalFlowInRegion(ofRectangle region){
		const Scalar& sc = sum(flow(toCv(region)));
		return ofVec2f(sc[1], sc[0]);
	}
	
    int FlowFarneback::getWidth() {
        return flow.cols;
    }
    int FlowFarneback::getHeight() {
        return flow.rows;
    }
    
	void FlowFarneback::drawFlow(ofRectangle rect, float x2, float y2){
		ofVec2f offset(rect.x,rect.y);
		ofVec2f scale(rect.width/flow.cols, rect.height/flow.rows);
		ofVec2f otherOffset(x2, y2);
		int stepSize = 4; //TODO: make class-level parameteric
		ofPushStyle();
		srand(0);
		for(int y = 0; y < flow.rows; y += stepSize) {
			for(int x = 0; x < flow.cols; x += stepSize) {
				ofVec2f cur = ofVec2f(x, y) * scale + offset;
				ofColor color = ofColor::fromHex(rand() % 0xffffff);
				ofLine(cur, getFlowPosition(x, y) * scale + offset + otherOffset);
			} 
		}
		ofPopStyle();
	}
}
