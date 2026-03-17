import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VerifyAccountDetails } from './verify-account-details';

describe('VerifyAccountDetails', () => {
  let component: VerifyAccountDetails;
  let fixture: ComponentFixture<VerifyAccountDetails>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VerifyAccountDetails],
    }).compileComponents();

    fixture = TestBed.createComponent(VerifyAccountDetails);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
