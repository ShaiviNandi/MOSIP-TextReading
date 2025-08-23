const { pipeline, env } = require('@xenova/transformers');
const fs = require('fs');
const path = require('path');

// Disable local model loading for faster startup
env.allowRemoteModels = true;

class FieldAutoCorrector {
    constructor() {
        this.classifier = null;
        // Common field mappings for form documents
        this.fieldMappings = {
            // Personal Information
            'name': ['nm', 'nme', 'full name', 'fullname', 'fname', 'first name', 'firstname'],
            'first_name': ['fname', 'first nm', 'frst name', 'firstname'],
            'last_name': ['lname', 'last nm', 'lst name', 'lastname', 'surname'],
            'email': ['eml', 'email address', 'e-mail', 'mail'],
            'phone': ['phne', 'phone number', 'phne nmer', 'contact', 'mobile', 'mob'],
            'address': ['addr', 'addres', 'home address', 'residence'],
            'date_of_birth': ['dob', 'birth date', 'birthdate', 'dt of birth'],
            'age': ['ag', 'years'],
            'gender': ['sex', 'gndr'],
            
            // Document specific
            'document_number': ['doc no', 'document no', 'id number', 'id no'],
            'issue_date': ['issued date', 'date issued', 'issue dt'],
            'expiry_date': ['exp date', 'expiration date', 'valid until'],
            'nationality': ['country', 'nation'],
            'place_of_birth': ['birth place', 'birthplace', 'pob'],
            
            // Form specific
            'application_id': ['app id', 'application number', 'ref no', 'reference'],
            'signature': ['sign', 'applicant signature'],
            'photo': ['photograph', 'image'],
            
            // Common typos and variations
            'registration_number': ['reg no', 'registration no', 'reg number'],
            'father_name': ['fathers name', 'father nm', 'guardian name'],
            'mother_name': ['mothers name', 'mother nm'],
            'occupation': ['job', 'profession', 'work'],
            'qualification': ['education', 'degree', 'qualification']
        };
    }

    // Initialize the text classification pipeline
    async initialize() {
        try {
            console.log('Loading text similarity model...');
            // Using a lightweight sentence similarity model
            this.classifier = await pipeline(
                'feature-extraction',
                'Xenova/all-MiniLM-L6-v2'
            );
            console.log('Model loaded successfully!');
        } catch (error) {
            console.error('Error loading model:', error);
            throw error;
        }
    }

    // Calculate similarity between two strings using embeddings
    async calculateSimilarity(text1, text2) {
        try {
            const embedding1 = await this.classifier(text1, { pooling: 'mean', normalize: true });
            const embedding2 = await this.classifier(text2, { pooling: 'mean', normalize: true });
            
            // Calculate cosine similarity
            const similarity = this.cosineSimilarity(embedding1.data, embedding2.data);
            return similarity;
        } catch (error) {
            console.log(`Fallback to Levenshtein for: ${text1} vs ${text2}`);
            return this.levenshteinSimilarity(text1, text2);
        }
    }

    // Cosine similarity calculation
    cosineSimilarity(a, b) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    // Fallback Levenshtein distance similarity
    levenshteinSimilarity(str1, str2) {
        const matrix = Array(str2.length + 1).fill().map(() => Array(str1.length + 1).fill(0));
        
        for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
        for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;
        
        for (let j = 1; j <= str2.length; j++) {
            for (let i = 1; i <= str1.length; i++) {
                const cost = str1[i - 1] === str2[j - 1] ? 0 : 1;
                matrix[j][i] = Math.min(
                    matrix[j - 1][i] + 1,
                    matrix[j][i - 1] + 1,
                    matrix[j - 1][i - 1] + cost
                );
            }
        }
        
        const maxLength = Math.max(str1.length, str2.length);
        return 1 - matrix[str2.length][str1.length] / maxLength;
    }

    // Find the best matching field name
    async findBestMatch(inputField) {
        const inputFieldLower = inputField.toLowerCase().trim();
        let bestMatch = inputField;
        let highestSimilarity = 0;
        const threshold = 0.6; // Minimum similarity threshold

        // First check direct mappings
        for (const [correctField, variations] of Object.entries(this.fieldMappings)) {
            for (const variation of variations) {
                if (inputFieldLower === variation.toLowerCase()) {
                    return correctField;
                }
            }
        }

        // Then use semantic similarity
        for (const [correctField, variations] of Object.entries(this.fieldMappings)) {
            // Check against the correct field name
            const similarity = await this.calculateSimilarity(inputFieldLower, correctField);
            if (similarity > highestSimilarity && similarity > threshold) {
                highestSimilarity = similarity;
                bestMatch = correctField;
            }

            // Check against variations
            for (const variation of variations) {
                const variationSimilarity = await this.calculateSimilarity(inputFieldLower, variation);
                if (variationSimilarity > highestSimilarity && variationSimilarity > threshold) {
                    highestSimilarity = variationSimilarity;
                    bestMatch = correctField;
                }
            }
        }

        return bestMatch;
    }

    // Process the JSON file and correct field names
    async correctFields(inputData) {
        const correctedData = {};
        const corrections = [];

        console.log('Processing fields for correction...');
        
        for (const [key, value] of Object.entries(inputData)) {
            const correctedKey = await this.findBestMatch(key);
            
            if (correctedKey !== key) {
                corrections.push({
                    original: key,
                    corrected: correctedKey,
                    value: value
                });
                console.log(`Corrected: "${key}" → "${correctedKey}"`);
            }
            
            correctedData[correctedKey] = value;
        }

        return {
            correctedData,
            corrections,
            summary: {
                totalFields: Object.keys(inputData).length,
                correctedFields: corrections.length,
                accuracy: `${corrections.length}/${Object.keys(inputData).length} fields corrected`
            }
        };
    }

    // Main processing function
    async processFile(inputPath, outputPath) {
        try {
            // Read input JSON
            const inputData = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
            console.log('Input data loaded:', Object.keys(inputData));

            // Process corrections
            const result = await this.correctFields(inputData);

            // Write corrected data to output file
            fs.writeFileSync(outputPath, JSON.stringify(result.correctedData, null, 2));

            // Log results
            console.log('\n=== CORRECTION SUMMARY ===');
            console.log(`Total fields processed: ${result.summary.totalFields}`);
            console.log(`Fields corrected: ${result.summary.correctedFields}`);
            
            if (result.corrections.length > 0) {
                console.log('\nCorrections made:');
                result.corrections.forEach(correction => {
                    console.log(`  • "${correction.original}" → "${correction.corrected}"`);
                });
            }

            console.log(`\nCorrected data saved to: ${outputPath}`);
            return result;

        } catch (error) {
            console.error('Error processing file:', error);
            throw error;
        }
    }
}

// Main execution function
async function main() {
    const corrector = new FieldAutoCorrector();
    
    try {
        await corrector.initialize();
        
        const inputPath = path.join(__dirname, 'input.json');
        const outputPath = path.join(__dirname, 'fieldcorrected.json');
        
        if (!fs.existsSync(inputPath)) {
            console.error('Input file not found. Please create input.json first.');
            return;
        }
        
        await corrector.processFile(inputPath, outputPath);
        
    } catch (error) {
        console.error('Application error:', error);
    }
}

// Run the application
if (require.main === module) {
    main();
}

module.exports = FieldAutoCorrector;