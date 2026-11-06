// Read factory calibration and depth units; never start a capture pipeline.
#include <librealsense2/rs.hpp>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <fstream>

static void intrinsics(const rs2_intrinsics &i) {
    std::cout << "{\"width\":" << i.width << ",\"height\":" << i.height
              << ",\"fx\":" << i.fx << ",\"fy\":" << i.fy
              << ",\"ppx\":" << i.ppx << ",\"ppy\":" << i.ppy
              << ",\"model\":" << int(i.model) << ",\"coeffs\":[";
    for(int k=0;k<5;k++) std::cout << (k?",":"") << i.coeffs[k];
    std::cout << "]}";
}
int main(int argc, char **argv) {
    try {
        if(argc!=2) throw std::runtime_error("expected right-wrist device serial");
        rs2::context context;
        for(auto device:context.query_devices()) {
            std::string serial=device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
            std::string port=device.get_info(RS2_CAMERA_INFO_PHYSICAL_PORT), usb_serial, path=port;
            while(!path.empty()) {
                std::ifstream file(path+"/serial");
                if(file && std::getline(file,usb_serial)) break;
                auto slash=path.rfind('/');
                if(slash==std::string::npos) break;
                path=path.substr(0,slash);
            }
            if(serial!=argv[1] && usb_serial!=argv[1]) continue;
            rs2::video_stream_profile depth, color;
            float scale=0;
            for(auto sensor:device.query_sensors()) {
                if(auto ds=sensor.as<rs2::depth_sensor>()) scale=ds.get_depth_scale();
                for(auto p:sensor.get_stream_profiles()) {
                    auto v=p.as<rs2::video_stream_profile>();
                    if(!v || v.width()!=640 || v.height()!=360 || v.fps()!=30) continue;
                    if(v.stream_type()==RS2_STREAM_DEPTH && v.format()==RS2_FORMAT_Z16) depth=v;
                    if(v.stream_type()==RS2_STREAM_COLOR) color=v;
                }
            }
            if(!depth || !color || scale<=0) throw std::runtime_error("640x360 factory profiles or depth scale unavailable");
            auto e=depth.get_extrinsics_to(color);
            std::cout << std::setprecision(10) << "{\"serial\":\"" << serial
                      << "\",\"scale_m\":" << scale << ",\"depth\":";
            intrinsics(depth.get_intrinsics()); std::cout << ",\"color\":";
            intrinsics(color.get_intrinsics()); std::cout << ",\"rotation_column_major\":[";
            for(int k=0;k<9;k++) std::cout << (k?",":"") << e.rotation[k];
            std::cout << "],\"translation_m\":[";
            for(int k=0;k<3;k++) std::cout << (k?",":"") << e.translation[k];
            std::cout << "]}" << std::endl;
            return 0;
        }
        throw std::runtime_error("right-wrist serial not found through librealsense");
    } catch(const std::exception &e) { std::cerr << e.what() << std::endl; return 1; }
}
