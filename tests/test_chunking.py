"""
Test the new deep chunking on the SALTO PDF already in Qdrant.
Run with:  python test_chunking.py
"""
import sys; sys.path.insert(0, ".")
from dotenv import load_dotenv; load_dotenv("config/.env")

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from src.core.chunker import chunk_markdown

print(f"Chunk settings: size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP} tokens")
print()

# Simulate a typical SALTO datasheet text (~2000 tokens)
sample_text = """
SALTO Systems CCVD20xx / CCVD40xx Keycard Datasheet

PRODUCT OVERVIEW
The SALTO CCVD20xx and CCVD40xx keycards are smart access control cards that
use the NXP MIFARE DESFire EV3 chip, certified to EAL5+ security standards.
They provide a secure, keyless building experience and are compatible with
all SALTO access control readers and systems.

CHIP SPECIFICATIONS
- Chip: NXP MIFARE DESFire EV3
- Security Certification: Common Criteria EAL5+
- Operating Frequency: 13.56 MHz
- Communication Standard: ISO/IEC 14443-A
- Data Retention: 10 years
- Write Endurance: 500,000 programming cycles
- Memory: 2 KB (CCVD20xx) / 4 KB (CCVD40xx)
- Anti-tear mechanism: Yes
- AES 128-bit encryption: Yes

PHYSICAL SPECIFICATIONS
- Format: ISO CR-80 (85.60 x 53.98 mm)
- Thickness: 0.84 mm (±0.05 mm)
- Material: PVC
- Printing: Offset or digital printing compatible
- Magnetic Stripe: Optional (HiCo or LoCo)
- Signature Panel: Optional
- Embossing: Optional

OPERATING CONDITIONS
- Operating Temperature: -20°C to +70°C
- Storage Temperature: -40°C to +85°C
- Humidity: 5% to 95% RH (non-condensing)

MODELS AVAILABLE
CCVD2001 - 2KB DESFire EV3, white blank card
CCVD2002 - 2KB DESFire EV3, custom printing
CCVD4001 - 4KB DESFire EV3, white blank card
CCVD4002 - 4KB DESFire EV3, custom printing

CONTACT INFORMATION
For more information about SALTO Systems products and solutions:
- Website: www.saltosystems.com
- Email: info@saltosystems.com
- SALTO Systems Headquarters: Oiartzun, Spain
- EMEA Office: London, UK
- APAC Office: Singapore
- India Contact: saltosystems.com/contact

CERTIFICATIONS
- CE marking
- FCC certification
- RoHS compliant
- REACH compliant
- ISO 9001:2015 certified manufacturing

WHERE TO BUY
SALTO products are available through authorized distributors worldwide.
Visit saltosystems.com/where-to-buy to find a distributor in your region.
For India: Contact SALTO Systems Asia Pacific office or visit their
website to find local authorized partners and distributors in major
Indian cities including Mumbai, Delhi, Bangalore, and Chennai.

SALTO Systems reserves the right to modify technical specifications,
designs, and performance without notice.
"""

chunks = chunk_markdown(sample_text, source_url="pdf_SALTO-test")
print(f"Input: {len(sample_text)} chars")
print(f"Output: {len(chunks)} chunks")
print()
for i, c in enumerate(chunks):
    header = f" [{c['context_header']}]" if c.get('context_header') else ""
    print(f"Chunk {i+1:2d}{header}: {len(c['text'])} chars | {c['text'][:80].strip()!r}...")
