function crack_diagnosis_app3
% =====================================================================
%  콘크리트 균열 진단 앱 (v3)
%   · 분류기: 본인 전처리(흑백+미디언+샤프닝)+ResNet-18 특징+RBF SVM (최고 성능)
%   · 선형 검출: 본인 전처리 기반 + 선형 성분 추출
%   · 상태등급: 국토안전관리원 A~E 기준(보내준 표) + 사선(전단)균열 예외
%   · 스케일: 이미지에서 두 점을 '먼저' 선으로 찍고 -> '그 다음' 길이(mm) 입력
%
%  실행:  >> crack_diagnosis_app3
%  필요: Image Processing Toolbox (+분류기 사용 시 Deep Learning/ResNet-18)
% =====================================================================
    REG = buildRegulations();

    st = struct('imgPath','', 'img',[], 'net',[], 'mask',[], 'report',[], ...
                'logLines',{{}}, 'mmPerPx',[], 'roi',[], ...
                'modelFile','crack_net.mat', 'dataRoot','');

    if isfile(st.modelFile)
        try, S = load(st.modelFile,'net'); if isfield(S,'net'), st.net = S.net; end
        catch, st.net = []; end
    end

    % ================= GUI =================
    fig = uifigure('Name','콘크리트 균열 진단 앱 (v3)','Position',[70 70 1150 720]);
    outer = uigridlayout(fig,[1 2]); outer.ColumnWidth = {380,'1x'}; outer.Padding=[10 10 10 10];

    left = uipanel(outer,'Title','설정 및 실행');
    lg = uigridlayout(left,[13 2]);
    lg.RowHeight = [repmat({'fit'},1,12), {'1x'}]; lg.ColumnWidth = {150,'1x'};
    lg.Padding=[10 10 10 10]; lg.RowSpacing=6;

    btnLoad = uibutton(lg,'Text','① 이미지 불러오기','ButtonPushedFcn',@(s,e)onLoadImage());
    btnLoad.Layout.Row=1; btnLoad.Layout.Column=[1 2];
    fileLbl = uilabel(lg,'Text','선택된 파일: (없음)','WordWrap','on','FontColor',[.3 .3 .3]);
    fileLbl.Layout.Row=2; fileLbl.Layout.Column=[1 2];

    envLbl = uilabel(lg,'Text','대기 노출조건'); envLbl.Layout.Row=3; envLbl.Layout.Column=1;
    envDrop = uidropdown(lg,'Items',{'건조 (dry)','습윤 (humid)','부식성 (corrosive)','심한 부식 (severe)'}, ...
        'ItemsData',{'dry','humid','corrosive','severe'},'Value','humid');
    envDrop.Layout.Row=3; envDrop.Layout.Column=2;

    btnCalib = uibutton(lg,'Text','② 두 점 찍기 → 길이 입력 (스케일)','ButtonPushedFcn',@(s,e)onCalibrate());
    btnCalib.Layout.Row=4; btnCalib.Layout.Column=[1 2];
    scaleLbl = uilabel(lg,'Text','스케일: 미설정 (보정 필요)','WordWrap','on','FontColor',[.6 .3 0]);
    scaleLbl.Layout.Row=5; scaleLbl.Layout.Column=[1 2];

    useChk = uicheckbox(lg,'Text','AI 분류기로 균열 유무 먼저 판정','Value',~isempty(st.net));
    useChk.Layout.Row=6; useChk.Layout.Column=[1 2];
    modelLbl = uilabel(lg,'Text',modelStatusText(st),'WordWrap','on','FontColor',[.3 .3 .3]);
    modelLbl.Layout.Row=7; modelLbl.Layout.Column=[1 2];

    btnFolder = uibutton(lg,'Text','학습용 데이터 폴더 선택','ButtonPushedFcn',@(s,e)onPickFolder());
    btnFolder.Layout.Row=8; btnFolder.Layout.Column=[1 2];
    btnTrain = uibutton(lg,'Text','분류기 학습(RBF SVM)','ButtonPushedFcn',@(s,e)onTrain());
    btnTrain.Layout.Row=9; btnTrain.Layout.Column=[1 2];

    btnRun = uibutton(lg,'Text','③ 진단 실행','FontWeight','bold', ...
        'BackgroundColor',[.2 .45 .85],'FontColor','w','ButtonPushedFcn',@(s,e)onRun());
    btnRun.Layout.Row=10; btnRun.Layout.Column=[1 2];
    btnPdf = uibutton(lg,'Text','④ 리포트(PDF) 저장','Enable','off','ButtonPushedFcn',@(s,e)onSavePdf());
    btnPdf.Layout.Row=11; btnPdf.Layout.Column=[1 2];

    logTitle = uilabel(lg,'Text','── 실행 로그 ──','FontWeight','bold');
    logTitle.Layout.Row=12; logTitle.Layout.Column=[1 2];
    logArea = uitextarea(lg,'Editable','off','Value',{''});
    logArea.Layout.Row=13; logArea.Layout.Column=[1 2];

    right = uipanel(outer,'Title','결과'); rg = uigridlayout(right,[2 2]);
    rg.RowHeight={'2x','1x'}; rg.ColumnWidth={'1x','1x'}; rg.Padding=[10 10 10 10];
    ax1=uiaxes(rg); ax1.Layout.Row=1; ax1.Layout.Column=1; title(ax1,'입력 이미지'); axis(ax1,'off');
    ax2=uiaxes(rg); ax2.Layout.Row=1; ax2.Layout.Column=2; title(ax2,'균열 검출(적색)'); axis(ax2,'off');
    resArea = uitextarea(rg,'Editable','off','Value',{'순서: ① 이미지 → ② 두 점 찍고 길이 입력 → ③ 진단 실행.'});
    resArea.Layout.Row=2; resArea.Layout.Column=[1 2];

    logMsg('앱 준비 완료.');

    % ================= 콜백 =================
    function onLoadImage()
        try
            [f,p]=uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif','이미지'},'진단할 이미지 선택');
            if isequal(f,0), logMsg('이미지 선택 취소.'); return; end
            img = imread(fullfile(p,f));
            st.imgPath=fullfile(p,f); st.img=img; st.mmPerPx=[]; clearROI();
            scaleLbl.Text='스케일: 미설정 (보정 필요)'; scaleLbl.FontColor=[.6 .3 0];
            fileLbl.Text=['선택된 파일: ' f];
            imshow(toRGB(img),'Parent',ax1); title(ax1,'입력 이미지');
            cla(ax2); title(ax2,'균열 검출(적색)'); axis(ax2,'off');
            btnPdf.Enable='off';
            resArea.Value={'이미지를 불러왔습니다. [② 두 점 찍기 → 길이 입력]을 누르세요.'};
            logMsg(sprintf('이미지 로드: %s (%dx%d)', f, size(img,2), size(img,1)));
        catch ME
            uialert(fig,['이미지를 읽을 수 없습니다.' newline ME.message],'이미지 오류');
        end
    end

    function onCalibrate()
        if isempty(st.img), uialert(fig,'먼저 이미지를 불러오세요.','이미지 없음'); return; end
        clearROI();
        imshow(toRGB(st.img),'Parent',ax1); title(ax1,'입력 이미지');
        logMsg('이미지에서 길이를 아는 두 점을 선으로 그으세요(드래그).');
        try
            roi = drawline(ax1,'Color',[1 1 0],'LineWidth',2);
        catch ME
            uialert(fig,['선분 그리기 실패: ' ME.message],'보정 오류'); return;
        end
        if ~isvalid(roi) || isempty(roi.Position) || size(roi.Position,1)<2
            logMsg('보정 취소.'); if isvalid(roi), delete(roi); end; return;
        end
        st.roi = roi;
        pos = roi.Position;
        pixDist = hypot(pos(2,1)-pos(1,1), pos(2,2)-pos(1,2));
        if pixDist < 1, uialert(fig,'선분이 너무 짧습니다. 다시 시도하세요.','보정 오류'); return; end

        % 두 점을 찍은 '다음' 길이 입력
        resp = inputdlg(sprintf('방금 그린 선분(%.1f px)의 실제 길이를 입력하세요 (mm):', pixDist), ...
                        '스케일 길이 입력', 1, {'100'});
        if isempty(resp), logMsg('길이 입력 취소.'); return; end
        realMM = str2double(resp{1});
        if ~(isfinite(realMM) && realMM>0)
            uialert(fig,'길이는 0보다 큰 숫자여야 합니다.','입력 오류'); return;
        end
        st.mmPerPx = realMM / pixDist;
        scaleLbl.Text = sprintf('스케일: %.4f mm/px  (선분 %.1f px = %.1f mm)', st.mmPerPx, pixDist, realMM);
        scaleLbl.FontColor=[0 .45 .2];
        logMsg(sprintf('스케일 보정 완료: %.4f mm/px', st.mmPerPx));
    end

    function onPickFolder()
        d = uigetdir('','학습용 데이터셋 폴더 선택');
        if isequal(d,0), return; end
        st.dataRoot=d; logMsg(['학습 데이터 폴더: ' d]);
    end

    function onTrain()
        if isempty(st.dataRoot) || ~isfolder(st.dataRoot)
            uialert(fig,'먼저 [학습용 데이터 폴더 선택]에서 유효한 폴더를 지정하세요.','폴더 없음'); return;
        end
        dlg = uiprogressdlg(fig,'Title','분류기 학습','Message','특징 추출 + RBF SVM 학습 중...','Indeterminate','on');
        c = onCleanup(@() close(dlg)); %#ok<NASGU>
        try
            net = trainBestClassifier(st.dataRoot, st.modelFile);
            st.net=net; useChk.Value=true; modelLbl.Text=modelStatusText(st);
            logMsg('분류기 학습/저장 완료 (RBF SVM)');
            uialert(fig,['학습 완료 → ' st.modelFile],'완료','Icon','success');
        catch ME
            logMsg(['학습 실패: ' ME.message]);
            uialert(fig,['학습 오류: ' ME.message],'학습 오류');
        end
    end

    function onRun()
        if isempty(st.img), uialert(fig,'먼저 이미지를 불러오세요.','이미지 없음'); return; end
        env = envDrop.Value; logMsg('── 진단 시작 ──');

        crackFlag = [];
        if useChk.Value
            if isempty(st.net), logMsg('분류기 없음 → AI 단계 건너뜀.');
            else
                crackFlag = classifyImage(st.net, st.img);
                logMsg(['분류기 판정: ' flagText(crackFlag)]);
            end
        end

        try, mask = segmentCrack(st.img);
        catch ME, logMsg(['세그멘테이션 실패: ' ME.message]); mask = false(size(st.img,1),size(st.img,2)); end
        st.mask = mask;

        [verdict, reason] = decideScene(st.img, mask, crackFlag);
        logMsg(['장면 판정: ' verdict ' — ' reason]);
        imshow(toRGB(st.img),'Parent',ax1); title(ax1,'입력 이미지');

        switch verdict
            case 'invalid'
                imshow(toRGB(st.img),'Parent',ax2); title(ax2,'(해당 없음)');
                resArea.Value = {'■ 판정 결과: 잘못된 이미지입니다.','', reason, '', ...
                    '콘크리트 표면 균열 사진인지 확인 후 다시 시도하세요.'};
                st.report = struct('verdict','invalid','verdictMsg',reason, ...
                    'img',st.img,'mask',mask,'fname',st.imgPath,'env',env);
                btnPdf.Enable='on'; logMsg('종료: 잘못된 이미지.');

            case 'nocrack'
                imshow(toRGB(st.img),'Parent',ax2); title(ax2,'(균열 없음)');
                resArea.Value = {'■ 판정 결과: 균열이 없습니다.','', reason, '', ...
                    sprintf('대기 노출조건: %s', env)};
                st.report = struct('verdict','nocrack','verdictMsg',reason, ...
                    'img',st.img,'mask',mask,'fname',st.imgPath,'env',env);
                btnPdf.Enable='on'; logMsg('종료: 균열 없음.');

            case 'crack'
                if isempty(st.mmPerPx)
                    uialert(fig,['균열이 검출되었습니다. 폭을 mm로 내려면 스케일이 필요합니다.' newline ...
                        '[② 두 점 찍기 → 길이 입력]을 먼저 실행하세요.'],'스케일 필요');
                    logMsg('진단 보류: 스케일 미설정.'); return;
                end
                [wMax,wMean,len,oriDeg] = measureCrack(mask, st.mmPerPx);
                [gIdx,gLabel,gState,gAction,isDiag] = gradeCrack(wMax, oriDeg, REG);
                [cLevel,cText] = assessCorrosion(wMax, env, REG);

                try, imshow(imoverlay(im2uint8(mat2gray(toRGB(st.img))), mask, [1 0 0]),'Parent',ax2);
                catch, imshow(toRGB(st.img),'Parent',ax2); end
                title(ax2,'균열 검출(적색)');

                lines = {
                    '■ 판정 결과: 균열이 검출되었습니다.'
                    sprintf('대기 노출조건 : %s  (허용폭 %.2f mm)', env, REG.allow.(env))
                    sprintf('스케일       : %.4f mm/px', st.mmPerPx)
                    ''
                    sprintf('최대폭 %.3f mm / 평균폭 %.3f mm / 길이 %.1f mm', wMax, wMean, len)
                    sprintf('균열 방향    : %.0f° %s', oriDeg, tern(isDiag,'(사선/전단)',''))
                    ''
                    sprintf('① 상태등급 : %s — %s', gLabel, gState)};
                if isDiag, lines{end+1} = '   ⚠ 사선(전단)균열 → 폭 무관하게 등급 상향(최소 D)'; end
                lines = [lines; {
                    sprintf('③ 부식위험 : %s — %s', cLevel, cText)
                    sprintf('④ 조치     : %s', gAction)}];
                resArea.Value = lines;

                st.report = struct('verdict','crack','img',st.img,'mask',mask,'fname',st.imgPath,'env',env, ...
                    'wMax',wMax,'wMean',wMean,'len',len,'oriDeg',oriDeg,'allowMM',REG.allow.(env), ...
                    'gIdx',gIdx,'gLabel',gLabel,'gState',gState,'gAction',gAction,'isDiag',isDiag, ...
                    'cLevel',cLevel,'cText',cText);
                btnPdf.Enable='on'; logMsg('진단 완료.');
        end
    end

    function onSavePdf()
        if isempty(st.report), uialert(fig,'먼저 진단을 실행하세요.','데이터 없음'); return; end
        try
            outPdf = makeReport(st.report);
            logMsg(['리포트 저장: ' outPdf]);
            uialert(fig,['리포트를 저장했습니다.' newline outPdf],'완료','Icon','success');
        catch ME
            uialert(fig,['리포트 저장 오류: ' ME.message],'PDF 오류');
        end
    end

    function clearROI(), if ~isempty(st.roi)&&isvalid(st.roi), delete(st.roi); end, st.roi=[]; end
    function logMsg(m)
        st.logLines{end+1}=char(m);
        if numel(st.logLines)>200, st.logLines=st.logLines(end-199:end); end
        logArea.Value=st.logLines; scroll(logArea,'bottom'); drawnow limitrate;
    end
end


% ===================== 로컬 함수 (스크립트와 동일 로직) ===============
function REG = buildRegulations()
    REG.allow = struct('dry',0.40,'humid',0.30,'corrosive',0.20,'severe',0.15);
    REG.gradeEdges  = [0.10 0.30 0.50 1.00];   % A|B|C|D|E 경계 [mm] (본 시스템 채택 기준)
    REG.gradeLabels = {'A (우수)','B (양호)','C (보통)','D (미흡)','E (불량)'};
    REG.gradeState  = {
        '균열 거의 없음 또는 0.1mm 미만 미세균열'
        '0.1mm 이상 0.3mm 미만 수준의 균열'
        '0.3mm 이상의 균열 또는 철근 피복 박리 징후'
        '구조적 결함으로 이어지는 대형 균열, 내하력 저하 우려'
        '심각한 구조 균열 및 변형으로 붕괴 위험 우려'};
    REG.gradeAction = {
        '구조적 영향 없음. 지속적 관찰'
        '경미한 결함. 표면처리 등 간단한 보수'
        '부재 성능 저하 방지를 위해 단면 복구·주입 보수'
        '긴급 보수·보강 필요. 사용 제한 여부 검토'
        '즉각 사용 금지. 개축 등 근본적 대책 필요'};
end

function net = trainBestClassifier(rootDir, modelFile)
    imds = imageDatastore(rootDir,'IncludeSubfolders',true, ...
        'FileExtensions',{'.jpg','.jpeg','.png'},'LabelSource','foldernames');
    folders = lower(string(imds.Labels)); bin = strings(size(folders));
    for i=1:numel(folders)
        f=folders(i);
        if contains(f,'non')||contains(f,'uncrack')||startsWith(f,'u'), bin(i)="noncrack";
        elseif contains(f,'crack')||startsWith(f,'c'), bin(i)="crack";
        else, bin(i)="noncrack"; end
    end
    imds.Labels = categorical(bin);
    counts = countEachLabel(imds); n = min([counts.Count; 2000]);
    imds = splitEachLabel(imds, n, 'randomized');
    imds.ReadFcn = @(fn) preprocessCrackImage(imread(fn));
    inSize=[224 224 3]; base=resnet18; layer='pool5';
    aug = augmentedImageDatastore(inSize, imds,'ColorPreprocessing','gray2rgb');
    X = activations(base, aug, layer,'OutputAs','rows','MiniBatchSize',64,'ExecutionEnvironment','auto');
    clf = fitcsvm(X, imds.Labels,'KernelFunction','rbf','KernelScale','auto','Standardize',true);
    net = struct('type','featureSVM','base',base,'layer',layer,'clf',clf,'inSize',inSize,'method','RBF SVM');
    save(modelFile,'net');
end

function isCrack = classifyImage(net, img)
    try
        if isstruct(net) && isfield(net,'base')
            g  = preprocessCrackImage(img);
            g3 = imresize(repmat(g,1,1,3), net.inSize(1:2));
            feat = activations(net.base, g3, net.layer,'OutputAs','rows');
            label = predict(net.clf, feat);
        else
            label = classify(net, imresize(img,[224 224]));
        end
        isCrack = (string(label)=="crack");
    catch
        isCrack = NaN;
    end
end

function outImg = preprocessCrackImage(inImg)
    if size(inImg,3)==3, grayImg=rgb2gray(inImg); else, grayImg=inImg; end
    filteredImg = medfilt2(grayImg,[3 3]);
    outImg = imsharpen(filteredImg,'Radius',2,'Amount',1.5);
end

function mask = segmentCrack(img)
    g = im2double(preprocessCrackImage(img));
    dark = mat2gray(imbothat(g, strel('disk',6)));
    t = graythresh(dark); bw = dark > max(t,0.15);
    bw = bwareaopen(bw,40); bw = imclose(bw, strel('disk',3));
    mask = false(size(bw));
    if ~any(bw(:)), return; end
    cc=bwconncomp(bw); stt=regionprops(cc,'MajorAxisLength','MinorAxisLength');
    dl=hypot(size(bw,1),size(bw,2));
    for k=1:cc.NumObjects
        maj=stt(k).MajorAxisLength; minr=max(stt(k).MinorAxisLength,1);
        if maj>=0.15*dl && (maj/minr)>=4, mask(cc.PixelIdxList{k})=true; end
    end
    if ~any(mask(:))
        bK=0;bM=0;
        for k=1:cc.NumObjects
            maj=stt(k).MajorAxisLength;
            if maj/max(stt(k).MinorAxisLength,1)>=3 && maj>bM, bM=maj;bK=k; end
        end
        if bK>0 && bM>=0.15*dl, mask(cc.PixelIdxList{bK})=true; end
    end
end

function [wMax,wMean,len,oriDeg] = measureCrack(mask, mmPerPx)
    wMax=0;wMean=0;len=0;oriDeg=NaN;
    if ~any(mask(:)), return; end
    skel=bwskel(mask); lenPx=nnz(skel);
    D=bwdist(~mask); wPx=2*D(skel); wPx=wPx(wPx>0);
    if isempty(wPx), wPx=2*max(D(:)); end
    wMax=max(wPx)*mmPerPx; wMean=mean(wPx)*mmPerPx; len=lenPx*mmPerPx;
    cc=bwconncomp(mask); s=regionprops(cc,'Orientation','MajorAxisLength');
    if ~isempty(s), [~,bi]=max([s.MajorAxisLength]); oriDeg=s(bi).Orientation; end
end

function [verdict,reason] = decideScene(img, mask, crackFlag)
    h=size(img,1); w=size(img,2); ratio=nnz(mask)/max(h*w,1);
    if h<64||w<64, verdict='invalid'; reason='이미지 해상도가 너무 낮습니다.'; return; end
    if ratio>0.35, verdict='invalid'; reason=sprintf('검출 영역 과다(%.0f%%) — 콘크리트 균열 이미지로 보기 어렵습니다.',ratio*100); return; end
    if islogical(crackFlag)&&isscalar(crackFlag)&&~crackFlag
        verdict='nocrack'; reason='분류기가 균열 없음으로 판정.'; return;
    end
    if ~any(mask(:)), verdict='nocrack'; reason='유효한 선형 균열이 검출되지 않음.'; return; end
    verdict='crack'; reason=sprintf('선형 균열 검출(영역 %.2f%%).',ratio*100);
end

function [idx,label,state,action,isDiag] = gradeCrack(wMM, oriDeg, REG)
    e=REG.gradeEdges;
    if     wMM<e(1), idx=1; elseif wMM<e(2), idx=2;
    elseif wMM<e(3), idx=3; elseif wMM<e(4), idx=4; else, idx=5; end
    isDiag = ~isnan(oriDeg) && abs(oriDeg)>=25 && abs(oriDeg)<=65;
    if isDiag, idx=max(idx,4); end
    label=REG.gradeLabels{idx}; state=REG.gradeState{idx}; action=REG.gradeAction{idx};
end

function [level,txt] = assessCorrosion(wMM, env, REG)
    allow=REG.allow.(env); ratio=wMM/allow;
    if     ratio<=0.5, level='낮음'; elseif ratio<=1.0, level='보통';
    elseif ratio<=1.5, level='높음'; else, level='매우 높음'; end
    if ratio>1.0, txt=sprintf('허용폭(%.2fmm,%s) 초과 -> 방수/보수 검토.',allow,env);
    else, txt=sprintf('허용폭(%.2fmm,%s) 이내 -> 정기 관찰.',allow,env); end
end

function outPdf = makeReport(R)
    imgRGB=toRGB(R.img);
    fig=figure('Color','w','Position',[100 100 980 660],'Visible','off');
    c=onCleanup(@() close(fig)); %#ok<NASGU>
    tl=tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');
    title(tl,'균열 진단 안전성 평가 리포트','FontWeight','bold','FontSize',15);

    nexttile; imshow(imgRGB); title('① 입력 이미지','FontWeight','bold');
    nexttile;
    if strcmp(R.verdict,'crack') && any(R.mask(:))
        try, imshow(imoverlay(im2uint8(mat2gray(imgRGB)),R.mask,[1 0 0])); catch, imshow(imgRGB); end
        title('② 균열 검출(적색)','FontWeight','bold');
    else, imshow(imgRGB); title('② 검출 없음','FontWeight','bold'); end

    if strcmp(R.verdict,'crack')
        nexttile; axis off;
        drawPanel('③ 정량 측정',{
            sprintf('최대 균열폭 : %.3f mm',R.wMax)
            sprintf('평균 균열폭 : %.3f mm',R.wMean)
            sprintf('균열 길이   : %.1f mm',R.len)
            sprintf('균열 방향   : %.0f° %s',R.oriDeg,tern(R.isDiag,'(사선/전단)',''))
            sprintf('허용 균열폭 : %.2f mm (%s)',R.allowMM,R.env)},[.1 .1 .6]);
        nexttile; axis off;
        cols={[0 .5 .2],[.3 .55 0],[.8 .65 0],[.85 .4 0],[.75 0 0]};
        lines={sprintf('① 상태등급 : %s',R.gLabel); sprintf('   %s',R.gState)};
        if R.isDiag, lines{end+1}='   ⚠ 사선(전단)균열 → 폭 무관 등급 상향'; end
        lines=[lines;{sprintf('③ 부식위험 : %s',R.cLevel); sprintf('④ 조치     : %s',R.gAction)}];
        drawPanel('④ 진단 (등급·부식·조치)',lines,cols{R.gIdx});
    else
        nexttile([1 2]); axis off;
        if strcmp(R.verdict,'nocrack'), head='판정 결과: 균열이 없습니다.'; col=[0 .5 .2];
        else, head='판정 결과: 잘못된 이미지입니다.'; col=[.75 0 0]; end
        drawPanel(head,{R.verdictMsg,'',sprintf('대기 노출조건: %s',R.env)},col);
    end

    annotation(fig,'textbox',[0.02 0.005 0.96 0.04], ...
        'String',sprintf('파일: %s  |  근거: 국토안전관리원 상태평가기준, KDS 14 20 30  |  ※ 현장 정밀점검을 대체하지 않음',shortPath(R.fname)), ...
        'EdgeColor','none','FontSize',8,'Color',[.3 .3 .3],'Interpreter','none');
    [~,b]=fileparts(R.fname); if isempty(b), b='result'; end
    outPdf=fullfile(pwd,sprintf('report_%s.pdf',b));
    exportgraphics(fig,outPdf,'ContentType','image','Resolution',200);
end

function drawPanel(ttl,lines,col)
    text(0,1,ttl,'FontWeight','bold','FontSize',12,'Color',col,'Units','normalized','VerticalAlignment','top');
    y=0.84;
    for i=1:numel(lines)
        text(0,y,lines{i},'FontSize',10,'Units','normalized','VerticalAlignment','top','Interpreter','none'); y=y-0.13;
    end
end

function rgb=toRGB(img), if size(img,3)==1, rgb=repmat(img,1,1,3); else, rgb=img; end, end
function s=tern(c,a,b), if c, s=a; else, s=b; end, end
function s=shortPath(p), if isempty(p), s='(없음)'; return; end, [~,n,e]=fileparts(p); s=[n e]; end
function s=modelStatusText(st)
    if isempty(st.net), s='분류기 상태: 없음 (영상처리만으로도 동작)';
    else, s='분류기 상태: 로드됨 (RBF SVM)'; end
end
function s=flagText(f)
    if isempty(f), s='미사용';
    elseif isnumeric(f)&&isscalar(f)&&isnan(f), s='판정 불가';
    else, s=tern(logical(f),'균열 있음','균열 없음'); end
end
