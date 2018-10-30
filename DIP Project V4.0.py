import numpy
import cv2
import matplotlib.pyplot as plt
import sys, os

def skullRemover(imgIn):

    rows = imgIn.shape[0]
    columns = imgIn.shape[1]

    skullImg = numpy.zeros((rows, columns), dtype=numpy.uint8)

    for r in range(rows):
        for c in range(columns):
            if imgIn[r][c] != 0:
                skullImg[r][c] = 255
            else:
                skullImg[r][c] = 0

    imgOut = cv2.connectedComponents(skullImg)[1]

    for r in range(rows):
        for c in range(columns):
            if imgOut[r][c] == 1:
                imgIn[r][c] = 0

def imageLabeller(img,labelImg,   type):

    if type == 2:
        skullRemover(labelImg)

    rows = labelImg.shape[0]
    columns = labelImg.shape[1]

    cpf = numpy.zeros((rows, columns), dtype=numpy.uint8)
    whiteMatter = numpy.zeros((rows, columns), dtype=numpy.uint8)
    greyMatter = numpy.zeros((rows, columns), dtype=numpy.uint8)
    colorLabel = numpy.zeros((rows, columns, 3), dtype=numpy.uint8)
    for r in range(rows):
        for c in range(columns):

            if labelImg[r][c] == 3 and type == 2:
                cpf[r][c] = 255
                colorLabel[r][c][1] = 255
            elif labelImg[r][c] == 2 and type == 2:
                greyMatter[r][c] = 255
                colorLabel[r][c][2] = 255
            elif labelImg[r][c] == 1 and type == 2:
                whiteMatter[r][c] = 255
                colorLabel[r][c][0] = 255
            elif labelImg[r][c] == 1 and type == 1:
                cpf[r][c] = 255
                colorLabel[r][c][1] = 255
            elif labelImg[r][c] == 2 and type == 1:
                greyMatter[r][c] = 255
                colorLabel[r][c][2] = 255
            elif labelImg[r][c] == 3 and type == 1:
                whiteMatter[r][c] = 255
                colorLabel[r][c][0] = 255
            else:
                colorLabel[r][c][:] = img[r][c]



    plt.figure(1)
    plt.gray()
    plot = plt.subplot(231)
    plot.axis("off")
    plot.imshow(img)
    plot.set_title('Original Image')

    plot = plt.subplot(232)
    plot.axis("off")
    plot.imshow(cv2.cvtColor(colorLabel, cv2.COLOR_BGR2RGB))
    plot.set_title('Segmented Image')

    plot = plt.subplot(233)
    plot.axis("off")
    plot.imshow(cpf)
    plot.set_title('CPF Areas')

    plot = plt.subplot(234)
    plot.axis("off")
    plot.imshow(whiteMatter)
    plot.set_title('White Matter Areas')

    plot = plt.subplot(235)
    plot.axis("off")
    plot.imshow(greyMatter)
    plot.set_title('Grey Matter Areas')
    plt.show()
    return colorLabel


def kmeansClusters(img, k, type):

    if type == 1:
        img = cv2.equalizeHist(img)

    sections = k
    rows = img.shape[0]
    columns = img.shape[1]
    marks = numpy.zeros((1, sections), dtype=numpy.uint8)

    for k in range(0, sections):
        marks[0][k] = k * 255 / (sections - 1)

    sumMat = numpy.zeros((1, sections), dtype=numpy.int)
    countMat = numpy.zeros((1, sections), dtype=numpy.int)
    flag = 1
    while (flag):
        labelImg = numpy.zeros((rows, columns), dtype=numpy.uint8)
        temp = numpy.zeros((1, sections), dtype=numpy.uint8)

        for r in range(rows):
            for c in range(columns):
                for k in range(0, sections):
                    diff = int(img[r][c]) - int(marks[0][k])
                    temp[0][k] = abs(diff)
                pos = 0
                minVal = temp.min()
                for k in range(0, sections):
                    if temp[0][k] == minVal:
                        pos = k
                labelImg[r][c] = pos

                sumMat[0][labelImg[r][c]] = sumMat[0][labelImg[r][c]] + img[r][c]
                countMat[0][labelImg[r][c]] = countMat[0][labelImg[r][c]] + 1

        checkMat = numpy.zeros((1, sections), dtype=numpy.int)
        flagMat = numpy.zeros((1, sections), dtype=numpy.int)
        for k in range(sections):
            checkMat[0][k] = marks[0][k]
            marks[0][k] = round(sumMat[0][k] / countMat[0][k])
            if checkMat[0][k] == marks[0][k]:
                flagMat[0][k] = 0
            else:
                flagMat[0][k] = 1

        flag = numpy.amin(flagMat)

    return labelImg


def segmentImage(type):

    #test with other images as well by changing the image name, check the folders for available images
    if type == 1:
        img = cv2.imread(os.path.dirname(os.path.abspath(sys.argv[0]))+ '\T1/t12.jpg', cv2.IMREAD_GRAYSCALE)
    elif type == 2:
        img = cv2.imread(os.path.dirname(os.path.abspath(sys.argv[0]))+ '\T2/t22.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        print('Input Argument Should be 1 for T1 images and 2 for T2 images')
        return

    labelImg = kmeansClusters(img, 4, type)

    labelImgClustered = imageLabeller(img, labelImg, type)

#Pass 1 to test on T1 images and 2 to test on T2 images
segmentImage(2)

