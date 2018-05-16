//
//  ViewController.swift
//  TensorflowIOSDemo
//
//  Created by Wiki on 2018/5/14.
//  Copyright © 2018年 Wiki. All rights reserved.
//

import UIKit
import CoreML
class ViewController: UIViewController {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var label: UILabel!
    override func viewDidLoad() {
        super.viewDidLoad()
        useCoreML2()
    }
    func useCoreML2(){
        let imagePath = Bundle.main.path(forResource: "test_image", ofType: "png")
        let imageData:UIImage = UIImage(contentsOfFile: imagePath!)!
        imageView.image = imageData
        let width = imageData.cgImage?.width
        let height = imageData.cgImage?.height
        
        let data:UnsafePointer<UInt8> = CFDataGetBytePtr(imageData.cgImage?.dataProvider?.data!)
        var image = Array<Int>()
        for i in 0...(width! * height! - 1){
            let postion = i*4
            // gaijin
            image.append(Int(data[postion]))
        }
        
        let mnist = Mnist()
        do{
            // print shape
            for i in 0...783{
                if(i % 28 == 0){
                    print("")
                }
                print("\(String(format: "% 2x", image[i]))",terminator: "")
            }
            print("")
            
            let array = try MLMultiArray(shape: [784], dataType: MLMultiArrayDataType.float32)
            for i in 0...(image.count-1) {
                let value = Double(image[i]) / 255.0
                array[i]  = NSNumber(floatLiteral: value)
            }
            let mnistInput = MnistInput(input__x_input__0: array)
            let result = try mnist.prediction(input: mnistInput)
            var text = ""
            for i in 0...(result.Softmax__0.count - 1){
                let item = result.Softmax__0[i]
                text += "\(i) possibility : \(String(format:"%.2f",item.floatValue))\n"
            }
            label.text = text
        }catch{
            print(error)
        }
    }
    
    func useCoreML(){
        let mnist = Mnist()
        do{
            let image = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,18,18,18,126,136,175,26,166,255,247,127,0,0,0,0,0,0,0,0,0,0,0,0,30,36,94,154,170,253,253,253,253,253,225,172,253,242,195,64,0,0,0,0,0,0,0,0,0,0,0,49,238,253,253,253,253,253,253,253,253,251,93,82,82,56,39,0,0,0,0,0,0,0,0,0,0,0,0,18,219,253,253,253,253,253,198,182,247,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,156,107,253,253,205,11,0,43,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,1,154,253,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,190,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,190,253,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,241,225,160,108,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,240,253,253,119,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,186,253,253,150,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,93,252,253,187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,249,253,249,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,130,183,253,253,207,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,148,229,253,253,253,250,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,114,221,253,253,253,253,201,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,66,213,253,253,253,253,198,81,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,171,219,253,253,253,253,195,80,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,172,226,253,253,253,253,244,133,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,136,253,253,253,212,135,132,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            // print shape
            for i in 0...783{
                if(i % 28 == 0){
                    print("")
                }
                print("\(String(format: "% 2x", image[i]))",terminator: "")
            }
            print("")
            
            let array = try MLMultiArray(shape: [784], dataType: MLMultiArrayDataType.float32)
            for i in 0...(image.count-1) {
                let value = Double(image[i]) / 255.0
                array[i]  = NSNumber(floatLiteral: value)
            }
            let mnistInput = MnistInput(input__x_input__0: array)
            let result = try mnist.prediction(input: mnistInput)
            for i in 0...(result.Softmax__0.count - 1){
                let item = result.Softmax__0[i]
                print("\(i) possibility : \(String(format:"%.2f",item.floatValue))")
            }
        }catch{
            print(error)
        }
    }
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
}




let imagePath = Bundle.main.path(forResource: "test_image", ofType: "png")
let imageData:UIImage = UIImage(contentsOfFile: imagePath!)!
imageView.image = imageData
let width = imageData.cgImage?.width
let height = imageData.cgImage?.height

let data:UnsafePointer<UInt8> = CFDataGetBytePtr(imageData.cgImage?.dataProvider?.data!)
var image = Array<Int>()
for i in 0...(width! * height! - 1){
    let postion = i*4
    // gaijin
    image.append(Int(data[postion]))
}

let mnist = Mnist()
do{
    // print shape
    for i in 0...783{
        if(i % 28 == 0){
            print("")
        }
        print("\(String(format: "% 2x", image[i]))",terminator: "")
    }
    print("")
    
    let array = try MLMultiArray(shape: [784], dataType: MLMultiArrayDataType.float32)
    for i in 0...(image.count-1) {
        let value = Double(image[i]) / 255.0
        array[i]  = NSNumber(floatLiteral: value)
    }
    let mnistInput = MnistInput(input__x_input__0: array)
    let result = try mnist.prediction(input: mnistInput)
    var text = ""
    for i in 0...(result.Softmax__0.count - 1){
        let item = result.Softmax__0[i]
        text += "\(i) possibility : \(String(format:"%.2f",item.floatValue))\n"
    }
    label.text = text
}catch{
    print(error)
}
