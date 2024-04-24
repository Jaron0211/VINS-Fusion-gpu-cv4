#include "enhancer.h"

float continue_beta = 0;
bool is_agc_init = false;
std::array<float,256> preview_pdf;
float lumi_error = 0;

cv::Mat adaptive_correction_mono(cv::Mat image0){
    
    using namespace cv;
    using namespace std;

    cv::Mat dstHist;
    int dims = 1;
    float hranges[] = {0,256};
    const float *ranges[] = {hranges};
    int size = 256;
    int channels = 0;

    cv::calcHist(&image0,1,&channels,cv::Mat(),dstHist,dims,&size,ranges);
    cv::Mat dstImage(size, size, CV_8U, Scalar(0));

    double minValue = 0;
    double maxValue = 0;
    cv::minMaxLoc(image0,&minValue, &maxValue, 0, 0);

    //This step is ref. from paper "Robust visual odometry based on image enhancement"
    //standardized
    float averge = 0;
    for(int i = 0; i<256; i++){
        float binValue = dstHist.at<float>(i);
        averge += binValue/255; 
    }

    float RM_error = 0;
    for(int i = 0; i<256; i++){
        float binValue = dstHist.at<float>(i);
        RM_error += pow((binValue - averge),2);
    }
    RM_error = sqrt(RM_error/255);

    float sum_of_SD_dst = 0;
    std::array<float,256> SD_dst;
    for(int i = 0; i<256 ;i++){
        float SD_value = (dstHist.at<float>(i)-averge)/RM_error;
        sum_of_SD_dst += SD_value;
        SD_dst[i] = SD_value;
    }

    //caculate the gamma(l)
    float gamma_value = 1;
    gamma_value = 1/(1 - sum_of_SD_dst);
    //std::cout << "gamma: " << gamma_value << std::endl;

    //gamma_correction
    cv::Scalar mean_scaler = cv::mean(image0);
    float img_Mean = mean_scaler.val[0];

    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_value) * 255.0);

    
    cv::Mat res0 = image0.clone();
    cv::LUT(image0, lookUpTable, res0);

    if(!image0.empty()){
        cv::imshow("image0",image0);
        cv::imshow("image0_correct",res0);
    }

    return res0;
    

}

std::pair<cv::Mat,cv::Mat> adaptive_correction_stereo(cv::Mat image0, cv::Mat image1){

    return std::make_pair(image0, image1);

} 

// THIS METHOD ORIGIN PROPOSED IN PAPER "Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution"
std::pair<cv::Mat,cv::Mat> adaptive_correction_stereo_agcwd(cv::Mat image0, cv::Mat image1){

    using namespace cv;
    using namespace std;

    cv::Mat dstHist;
    int dims = 1;
    float hranges[] = {0,256};
    const float *ranges[] = {hranges};
    int size = 256;
    int channels = 0;

    cv::calcHist(&image0,1,&channels,cv::Mat(),dstHist,dims,&size,ranges);

    float M = image0.size().width ;
    float N = image0.size().height ; 
    float pixel_size = (M*N);

    std::array<float,256> pdf;
    for(int i = 0; i<256; i++){
        pdf[i] = (dstHist.at<float>(i))/pixel_size;
    }

    //Add lumination robust adaptive
    float max_lumi_error = 30;
    if(!is_agc_init){    
        preview_pdf = pdf;
        is_agc_init = true;
    }else{
        float last_sum = 0;
        float this_sum = 0;
        for(int i = 0; i<256; i++){
            last_sum = last_sum + preview_pdf[i]*i ;
            this_sum = this_sum + pdf[i]*i;
        }
        lumi_error = (lumi_error + (last_sum - this_sum));
        lumi_error = lumi_error > max_lumi_error ? max_lumi_error : lumi_error;
        lumi_error = lumi_error < -max_lumi_error ? -max_lumi_error : lumi_error;

        preview_pdf = pdf;
    }

    std::array<float,256> pdfw;
    float* pdf_max = std::max_element(pdf.begin(),pdf.end());
    float* pdf_min = std::min_element(pdf.begin(),pdf.end());
    float alpha = 1 ;

    for(int i = 0; i<256; i++){
        pdfw[i] = *pdf_max * pow(((pdf[i] - *pdf_min)/(*pdf_max - *pdf_min)),alpha);
    }

    std::array<float,256> cdf;
    for(int i = 0; i<256; i++){
        float sum = 0.0;
        for (int j = 0; j<=i ;j++){
            sum += float(pdf[j]);
        }
        cdf[i] = sum;
    }

    std::array<float,256> cdfw;
    float sum_of_pdfw = 0;
    for(int i = 0; i<256; i++){
        sum_of_pdfw += pdfw[i];
    }

    for(int i = 0; i<256; i++){
        float sum = 0.0;

        for (int j = 0; j<=i ;j++){
            sum += float(pdfw[j]);
        }
        cdfw[i] = sum/sum_of_pdfw;
    }

    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    float threshold = 1;
    std::array<float,256> enhance_array;
    for( int i = 0; i < 256; ++i){
        float gamma = 1-cdfw[i];
        gamma = std::max(gamma,threshold);
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
        enhance_array[i] = p[i];
    }

    cv::Mat dstImage(size, size, CV_8U, Scalar(0));
    float* cdfw_max = std::max_element(enhance_array.begin(),enhance_array.end());
    for(int i = 0; i < 256; i++)
    {
        float binValue = enhance_array[i];   
        int intensity = cvRound(binValue * (size-1) / *cdfw_max );
        cv::line(dstImage,cv::Point(i, size - intensity),cv::Point(i, size - 1 ),Scalar(255));
    }
    cv::imshow("smoothed gamma curve", dstImage);

    cv::Mat res0 ;
    cv::Mat res0_ir ;
    cv::LUT(image0, lookUpTable, res0);
    res0.convertTo(res0,-1,1 + (lumi_error/50),0);

    cv::Mat res1 ;
    cv::LUT(image1, lookUpTable, res1);
    res1.convertTo(res1,-1,1 + (lumi_error/50),0);

    if(!image0.empty()){
        cv::imshow("image0",image0);
    }
    if(!image1.empty()){
        cv::imshow("image1",image1);
    }

    cv::Mat imTrack;
    cv::hconcat(image0, res0_ir, imTrack);
    cv::imshow("Before and After", imTrack);

    cv::Mat after_dstHist;
    cv::calcHist(&res0,1,&channels,cv::Mat(),after_dstHist,dims,&size,ranges);
    cv::Mat after_dstImage(size, size, CV_8U, Scalar(0));

    double minValue = 0;
    double maxValue = 0;
    cv::minMaxLoc(after_dstHist,&minValue, &maxValue, NULL, NULL);

    for(int i = 0; i < 256; i++)
    {
        float binValue = after_dstHist.at<float>(i);   
        
        int intensity = cvRound(binValue * (size-1) /  maxValue);
        cv::line(after_dstImage,cv::Point(i, size - intensity),cv::Point(i, size - 1 ),Scalar(255));
    }
    cv::imshow("一维直方图", after_dstImage);

    cv::waitKey(1);
    return std::make_pair(res0, res1);

} 

//traditional Histogram Equalization
std::pair<cv::Mat,cv::Mat> adaptive_correction_stereo_THE(cv::Mat image0, cv::Mat image1){

    using namespace cv;
    using namespace std;

    cv::Mat dstHist;
    int dims = 1;
    float hranges[] = {0,256};
    const float *ranges[] = {hranges};
    int size = 256;
    int channels = 0;

    cv::calcHist(&image0,1,&channels,cv::Mat(),dstHist,dims,&size,ranges);

    float M = image0.size().width ;
    float N = image0.size().height ; 
    float pixel_size = (M*N);

    std::array<float,256> pdf;
    for(int i = 0; i<256; i++){
        pdf[i] = (dstHist.at<float>(i))/pixel_size;
    }

    std::array<float,256> cdf;
    for(int i = 0; i<256; i++){
        for (int j = 0; j<=i ;j++){
            cdf[i] += pdf[j];
        }
    }

    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(cdf[i]*255.0);

    cv::Mat res0 = image0.clone();
    cv::LUT(image0, lookUpTable, res0);

    cv::Mat res1 = image1.clone();
    cv::LUT(image1, lookUpTable, res1);


}

// pratice adaptive gamma correction
std::pair<cv::Mat,cv::Mat> adaptive_correction_stereo_old(cv::Mat image0, cv::Mat image1){
    
    using namespace cv;
    using namespace std;

    cv::Mat dstHist;
    int dims = 1;
    float hranges[] = {0,256};
    const float *ranges[] = {hranges};
    int size = 256;
    int channels = 0;

    cv::calcHist(&image0,1,&channels,cv::Mat(),dstHist,dims,&size,ranges);
    cv::Mat dstImage(size, size, CV_8U, Scalar(0));

    double minValue = 0;
    double maxValue = 0;
    cv::minMaxLoc(image0,&minValue, &maxValue, 0, 0);

    //This step is ref. from paper "Robust visual odometry based on image enhancement"
    //standardized
    float averge = 0;
    for(int i = 0; i<256; i++){
        float binValue = dstHist.at<float>(i);
        averge += binValue/255; 
    }

    float RM_error = 0;
    for(int i = 0; i<256; i++){
        float binValue = dstHist.at<float>(i);
        RM_error += pow((binValue - averge),2);
    }
    RM_error = sqrt(RM_error/255);

    float sum_of_SD_dst = 0;
    std::array<float,256> SD_dst;
    for(int i = 0; i<256 ;i++){
        float SD_value = (dstHist.at<float>(i)-averge)/RM_error;
        sum_of_SD_dst += SD_value;
        SD_dst[i] = SD_value;
    }

    //caculate the gamma(l)
    float gamma_value = 1;
    gamma_value = 1/(1 - sum_of_SD_dst);

    //gamma_correction
    cv::Scalar mean_scaler = cv::mean(image0);
    float img_Mean = mean_scaler.val[0];

    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i)
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_value) * 255.0);

    //histogram 
    cv::Mat res0 = image0.clone();
    cv::LUT(image0, lookUpTable, res0);

    cv::Mat res1 = image1.clone();
    cv::LUT(image1, lookUpTable, res1);

    if(!image0.empty()){
        cv::imshow("image0",image0);
        //cv::imshow("image0_correct",res0);
    }
    if(!image1.empty()){
        cv::imshow("image1",image1);
        //cv::imshow("image1_correct",res1);
    }

    cv::waitKey(1);
    return std::make_pair(res0, res1);
}