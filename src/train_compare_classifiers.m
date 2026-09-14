%% =====================================================================
%  여러 분류기 성능 비교 + 최고 성능 분류기 저장
%  (ResNet-18 특징 추출 1회 -> 여러 분류기 학습/평가 -> 최고 모델 저장)
%
%  저장 형식은 기존 crack_diagnosis / 앱의 classifyImage와 호환됩니다.
%  실행:  >> train_compare_classifiers
% =====================================================================
clc; clear; rng(0);   % rng(0): 재현성

%% --------------------------- 설정 -----------------------------------
DATA_ROOT  = fullfile(pwd, 'dataset');      % SDNET2018 폴더 경로로 바꿔서 사용
MODEL_FILE = fullfile(pwd, 'crack_net.mat'); % 학습된 모델 저장 위치
PER_CLASS  = 2000;     % 클래스당 사용 이미지 수 (1:1 균형). 빠르게:1000 / 정확히:3000
HOLDOUT    = 0.30;     % 테스트셋 비율

%% --------------------- 1) 데이터 로드 + 라벨 ------------------------
imds = imageDatastore(DATA_ROOT, 'IncludeSubfolders', true, ...
    'FileExtensions', {'.jpg','.jpeg','.png'}, 'LabelSource', 'foldernames');

folders = lower(string(imds.Labels));
bin = strings(size(folders));
for i = 1:numel(folders)
    f = folders(i);
    if contains(f,'non') || contains(f,'uncrack') || startsWith(f,'u')
        bin(i) = "noncrack";
    elseif contains(f,'crack') || startsWith(f,'c')
        bin(i) = "crack";
    else
        bin(i) = "noncrack";
    end
end
imds.Labels = categorical(bin);
if numel(categories(imds.Labels)) < 2
    error('crack/noncrack 2개 클래스를 찾지 못했습니다. 폴더 구조를 확인하세요.');
end

% 불균형 보정: 적은 쪽에 맞춰 1:1 언더샘플링 (상한 PER_CLASS)
counts = countEachLabel(imds);
n = min([counts.Count; PER_CLASS]);
imds = splitEachLabel(imds, n, 'randomized');
fprintf('사용 데이터(클래스당 %d장):\n', n); disp(countEachLabel(imds));


%% --------------------- 2) ResNet-18 특징 추출 -----------------------
inSize = [224 224 3]; base = resnet18; layer = 'pool5';   % ResNet-18 표준 입력
% 디스크에서 읽는 즉시 전처리(흑백+미디언+샤프닝)를 적용한다.
% 학습과 추론이 반드시 같은 전처리를 거쳐야 하므로 앱의 preprocessCrackImage와 동일해야 한다.
imds.ReadFcn = @(filename) preprocessCrackImage(imread(filename));

aug = augmentedImageDatastore(inSize, imds, 'ColorPreprocessing', 'gray2rgb');
fprintf('특징 추출 중... (이미지당 1회 순전파)\n');

X = activations(base, aug, layer, ...
    'MiniBatchSize', 64, ...
    'ExecutionEnvironment', 'auto', ...   % GPU 있으면 자동 사용
    'OutputAs', 'rows');                  % 224 입력이므로 rows 사용 가능 -> [N x 512]
Y = imds.Labels;
%% --------------------- 3) train/test 분할 (계층적) -------------------
cv  = cvpartition(Y, 'HoldOut', HOLDOUT);
Xtr = X(training(cv), :);  Ytr = Y(training(cv));
Xte = X(test(cv), :);      Yte = Y(test(cv));

%% --------------------- 4) 후보 분류기 정의 --------------------------
% {이름, @(X,Y) 학습함수} — 같은 함수를 전체 데이터 재학습에도 재사용
makers = {
  '선형 SVM',        @(Xa,Ya) fitcsvm(Xa, Ya, 'KernelFunction','linear', 'Standardize',true);
  'RBF SVM',         @(Xa,Ya) fitcsvm(Xa, Ya, 'KernelFunction','rbf', 'KernelScale','auto', 'Standardize',true);
  'k-NN (k=5)',      @(Xa,Ya) fitcknn(Xa, Ya, 'NumNeighbors',5, 'Standardize',true);
  '랜덤포레스트',    @(Xa,Ya) fitcensemble(Xa, Ya, 'Method','Bag', 'NumLearningCycles',100, 'Learners','tree');
  '부스팅 트리',     @(Xa,Ya) fitcensemble(Xa, Ya, 'Method','LogitBoost', 'NumLearningCycles',100, 'Learners','tree');
  'LDA',             @(Xa,Ya) fitcdiscr(Xa, Ya, 'DiscrimType','pseudoLinear');
};

%% --------------------- 5) 학습 + 평가 -------------------------------
nM = size(makers, 1);
names = strings(nM,1); acc = nan(nM,1); bacc = nan(nM,1);
f1 = nan(nM,1); tt = nan(nM,1); trained = cell(nM,1);

fprintf('\n분류기별 학습/평가...\n');
for i = 1:nM
    names(i) = makers{i,1};
    try
        t0 = tic;
        mdl = makers{i,2}(Xtr, Ytr);
        tt(i) = toc(t0);
        pred = predict(mdl, Xte);
        [acc(i), bacc(i), f1(i)] = binMetrics(Yte, pred, "crack");
        trained{i} = mdl;
        fprintf('  %-14s  정확도 %.3f | 균형정확도 %.3f | F1 %.3f | %.1fs\n', ...
                names(i), acc(i), bacc(i), f1(i), tt(i));
    catch ME
        fprintf('  %-14s  학습 실패: %s\n', names(i), ME.message);
    end
end

%% --------------------- 6) 결과 표 -----------------------------------
T = table(names, acc, bacc, f1, tt, ...
    'VariableNames', {'분류기','정확도','균형정확도','F1','학습시간_s'});
T = sortrows(T, '균형정확도', 'descend');
fprintf('\n===== 분류기 성능 비교 (테스트셋) =====\n');
disp(T);

%% --------------------- 7) 최고 분류기 선택 --------------------------
score = bacc + 1e-3*f1;          % 균형정확도 우선, F1 타이브레이크
score(isnan(score)) = -Inf;
[~, bestIdx] = max(score);
bestName = names(bestIdx);
fprintf('★ 선택된 최고 분류기: %s (균형정확도 %.3f, F1 %.3f)\n', ...
        bestName, bacc(bestIdx), f1(bestIdx));

% 혼동행렬 시각화(가능한 경우)
try
    figure('Name','최고 분류기 혼동행렬');
    confusionchart(Yte, predict(trained{bestIdx}, Xte));
    title(sprintf('%s — 테스트셋 혼동행렬', bestName));
catch
end

%% --------------------- 8) 전체 데이터로 재학습 후 저장 --------------
fprintf('전체 균형데이터로 최고 분류기 재학습 중...\n');
finalClf = makers{bestIdx,2}(X, Y);

net = struct('type','featureSVM', ...  % classifyImage 호환 태그(내부 추론은 predict 동일)
             'base', base, 'layer', layer, 'clf', finalClf, 'inSize', inSize, ...
             'method', char(bestName), 'metrics', T);
save(MODEL_FILE, 'net');
fprintf('저장 완료 -> %s  (best = %s)\n', MODEL_FILE, bestName);


%% ===================== 로컬 함수 ====================================
function [acc, bacc, f1] = binMetrics(ytrue, ypred, posClass)
% 이진 분류 지표. posClass를 양성(여기선 "crack")으로 둠.
    yt = (string(ytrue) == posClass);
    yp = (string(ypred) == posClass);
    TP = sum(yt & yp);  FP = sum(~yt & yp);
    FN = sum(yt & ~yp); TN = sum(~yt & ~yp);
    acc    = (TP + TN) / max(numel(yt), 1);
    recall = TP / max(TP + FN, 1);     % 민감도(균열 검출률)
    spec   = TN / max(TN + FP, 1);     % 특이도
    bacc   = (recall + spec) / 2;      % 균형정확도 (불균형에 강함)
    prec   = TP / max(TP + FP, 1);
    f1     = 2 * prec * recall / max(prec + recall, eps);
end


%% C추가함수
function outImg = preprocessCrackImage(inImg)
% 1. 색상 노이즈 무시: 이미지를 흑백으로 변환하여 순수한 '형태'와 '명암'만 남김
if size(inImg, 3) == 3
    grayImg = rgb2gray(inImg);
else
    grayImg = inImg;
end

% 2. 점 노이즈 제거: 3x3 미디언(중간값) 필터 적용
% 주변 픽셀과 튀는 '점' 형태의 노이즈는 지워지고, 길게 이어진 '선(균열)'은 보존됨
filteredImg = medfilt2(grayImg, [3 3]);

% 3. 선형 특징 강조: 점이 사라진 후 남은 직선 형태의 균열을 더 뚜렷하게 샤프닝
outImg = imsharpen(filteredImg, 'Radius', 2, 'Amount', 1.5);
end