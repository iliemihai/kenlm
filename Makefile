# Makefile to install CC-Net and train the LMs.
# `make` or `make help` to get some help.

# Arguments:
lang?=ro
process?=8
servers?=0

# Experiment config
NDOC_FOR_LM=1_000_000
VOCAB_SIZE=64000

# Static resources, scripts, ...
KENLM=./bin/lmplz
KENLM_BUILD_BINARY=./bin/build_binary
SPM_TRAIN=./bin/spm_train
SPM_ENCODE=./bin/spm_encode

# DISTRIBUTE will run locally, or on slurm if "servers" is set.
DISTRIBUTE=xargs -L1 -P $(process)
ifneq ($(servers), 0)
	DISTRIBUTE=xargs -L1 -P $(servers) srun -t 240 --mem 5000
endif

help:
	# Show help
	grep -i -A1 '^[a-z0-9_]*:' Makefile

install: bin/lid.bin $(KENLM) $(SPM_TRAIN)
	# Installs dependencies.
	@if [ -f "data" ]; then\
		echo "Please create/simlink a 'data' directory.";\
	fi
	@if ! python -c "from cc_net import __main__" 2> /dev/null; then\
		pip install . ;\
	fi
	echo " --> All dependencies looks good !"


lm: data/lm_sp/$(lang).sp.model data/lm_sp/$(lang).arpa.bin
	echo "AVEM 1"
	# Computes a 5-gram LM for the given language -> make lang=it lm
	# Restricted to the first NDOC_FOR_LM documents

sp: data/lm_sp/$(lang).sp.model
	echo "AVEM 2"
	# Train a sentence piece model on Wikipedia -> make lang=it sp

get_lang = $(firstword $(subst ., ,$1))


%.arpa.bin: %.arpa
	echo "AVEM 3"
	# Compress a learned LM to a binary format.
	$(KENLM_BUILD_BINARY) $< $@

%.vocab.txt: %.txt
	echo "AVEM 4"
	# Extracts the vocabulary of a corpus.
	# Restricted to the first NDOC_FOR_LM documents and VOCAB_SIZE top words.
	cat $< | tr ' ' '\n' | sort | uniq -c | sort -rn > $@.tmp_sort
	head -$(VOCAB_SIZE) $@.tmp_sort | sed "s/^ *//" | cut -d' ' -f2 > $@
	rm $@.tmp*
	echo Extracted `wc -l $@` words

data/lm_sp/%.arpa: data/cirrus/sp/%.opening.txt
	echo "AVEM 5" 
	mkdir -p $(@D)
	$(KENLM) -o 5 -S 8G -T /tmp --vocab_estimate $(VOCAB_SIZE)  --discount_fallback \
        < data/wiki_enc/ro.txt > $@

data/lm_sp/%.sp.model: data/wiki.txt
	echo "AVEM 6"
	mkdir -p $(@D)
	$(SPM_TRAIN) --input=$< \
		--vocab_size=$(VOCAB_SIZE) --hard_vocab_limit \
		--character_coverage=0.9995 \
		--model_type=unigram \
		--model_prefix=$(basename $@) \
	|| echo "WARNING: Corpus is too small, will train smaller model" && \
	$(SPM_TRAIN) --input=$< \
		--vocab_size=$(VOCAB_SIZE) \
		--character_coverage=0.9995 \
		--model_type=unigram \
		--model_prefix=$(basename $@)

	echo "Trained SentencePiece model with `wc -l $(basename $@).vocab` pieces"


data/cirrus/sp/%.opening.txt: data/cirrus/gz/%.json.gz  data/lm_sp/%.sp.model
	echo "AVEM 7" $@ $<
	$(SPM_ENCODE) \
		--model=$(word 2,$^) \
		--output_format=piece \
		data/wiki.txt \
	        > data/wiki_enc/ro.txt
			#< <(python get_wiki_cirrus.py opening --file $< --n_docs $(NDOC_FOR_LM)) \
			#> $@

data/cirrus/txt/%.opening.txt: data/cirrus/gz/%.json.gz
	echo "AVEM 8"
	#python get_wiki_cirrus.py opening \
	#	--n_docs $(NDOC_FOR_LM) \
	#	--file $< --output $@

data/cirrus/gz/%.json.gz:
	echo "AVEM 9"
	#mkdir $(@D)
	#python get_wiki_cirrus.py dl --lang $(call get_lang,$(@F)) --output_dir $(@D)

clean:
	# Remove intemediary files, dataset, third_party sources
	# We don't need the vocab files nor the text version of the LM.
	rm -r data/cirrus
	rm -r data/lm_sp/*.arpa data/lm_sp/*.vocab
	rm -r third_party

# Installation
bin/lid.bin:
	# DL languages id from Fasttext releases.
	mkdir -p $(@D)
	wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O $@

third_party/kenlm:
	# Download kenlm sources: https://kheafield.com/code/kenlm/"
	mkdir -p $(@D)
	wget -O - https://kheafield.com/code/kenlm.tar.gz | tar -xz -C $(@D)

bin/lmplz: third_party/kenlm
	# Compiles kenlm binaries
	mkdir -p $(@D)
	mkdir -p $</build
	(cd $</build && cmake ..)
	make -C $</build -j2
	mv $</build/bin/lmplz $</build/bin/build_binary $(@D)

third_party/sentencepiece:
	# Download sentencepiece sources: https://github.com/google/sentencepiece
	mkdir -p $(@D)
	wget -c -O $(@D)/sentencepiece.zip https://github.com/google/sentencepiece/archive/v0.1.83.zip
	unzip -o -d $(@D) $(@D)/sentencepiece.zip
	rm $(@D)/sentencepiece.zip
	# remove the version id from the folder name
	mv $(@D)/sentencepiece-* $@

bin/spm_train: third_party/sentencepiece
	# Compiles sentencepiece binaries
	mkdir -p $(@D)
	mkdir -p $</build
	(cd $</build && cmake ..)
	make -C $</build -j2
	mv $</build/src/spm_train $</build/src/spm_encode $(@D)
	# Installed SentencePiece locally to install globally do:
	# $ cd $</build
	# $ sudo make install
	# $ sudo ldconfig -v


